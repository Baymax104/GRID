"""观察真实 beam frontier，不改变 processor 返回的解码分数。"""

import torch
from transformers import LogitsProcessor


class PathTraceProcessor(LogitsProcessor):
    def __init__(self, processor, target_ids):
        self.processor = processor
        self.targets = target_ids.detach()
        self.frontiers = []
        self.branches = []

    def __call__(self, input_ids, scores):
        output = self.processor(input_ids, scores)
        depth, gen, content, valid = self.processor.distributions(input_ids, scores)
        if depth != len(self.frontiers):
            raise ValueError("Path trace requires one processor call per SID depth.")
        batch, h = self.targets.shape
        beams = len(input_ids) // batch
        offsets = 1 + torch.arange(depth, device=input_ids.device) * self.processor.k
        parents = (input_ids[:, 1:] - offsets).reshape(batch, beams, depth)
        self.frontiers.append(parents.detach().cpu())
        matches = (parents == self.targets[:, None, :depth]).all(-1)
        present = matches.any(-1)
        parent_index = matches.long().argmax(-1)
        row = torch.arange(batch, device=scores.device) * beams + parent_index
        token = self.targets[:, depth].long()
        start = 1 + depth * self.processor.k
        mixed = output[:, start : start + self.processor.k]
        selected = mixed[row, token]
        branch = {
            "target_parent_present": present,
            "target_generation_log_prob": gen[row, token],
            "target_content_log_prob": content[row, token],
            "target_mixed_log_prob": selected,
            # 此排名只比较同一父节点的合法子分支，分数相等时共享排名。
            "target_branch_rank": ((mixed[row] > selected[:, None]) & valid[row]).sum(-1) + 1,
            "legal_child_count": valid[row].sum(-1),
        }
        for name in tuple(branch):
            if name.endswith("log_prob"):
                branch[name] = branch[name].masked_fill(~present, torch.nan)
            elif name != "target_parent_present":
                branch[name] = branch[name].masked_fill(~present, 0)
        self.branches.append({k: v.detach().cpu() for k, v in branch.items()})
        return output

    def finish(self, final_sids):
        batch, h = self.targets.shape
        if len(self.frontiers) != h:
            raise ValueError("Path trace has incomplete decoding depths.")
        final = final_sids.detach().cpu().reshape(batch, -1, h)
        beams = final.shape[1]
        prefixes = final.new_full((batch, h, beams, h), -1)
        for depth in range(1, h + 1):
            frontier = final if depth == h else self.frontiers[depth]
            prefixes[:, depth - 1, :, :depth] = frontier
        targets = self.targets.cpu()
        survived = torch.stack([
            (prefixes[:, d, :, :d + 1] == targets[:, None, :d + 1]).all(-1).any(-1)
            for d in range(h)
        ], dim=1)
        failed = ~survived
        first = torch.where(failed.any(-1), failed.long().argmax(-1) + 1, -1)
        return {
            "beam_prefixes": prefixes,
            "target_prefix_survived": survived,
            "first_failure_depth": first,
            **{name: torch.stack([step[name] for step in self.branches], dim=1)
               for name in self.branches[0]},
        }
