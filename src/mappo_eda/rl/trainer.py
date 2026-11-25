"""MAPPO optimizer independent of the OpenROAD environment."""
from __future__ import annotations
import torch
from torch import Tensor, nn
from .buffer import RolloutBatch
from .losses import masked_mean, ppo_loss, value_loss

class MAPPOTrainer:
    def __init__(self, policy, lr=.0003, clip=.2, value_coef=.5, entropy_coef=0., max_grad_norm=.5, device="cpu"):
        self.device=torch.device(device); self.policy=policy.to(self.device); self.optimizer=torch.optim.Adam(self.policy.parameters(),lr=lr)
        self.clip=clip; self.value_coef=value_coef; self.entropy_coef=entropy_coef; self.max_grad_norm=max_grad_norm
    def evaluate_actor(self, stage, actor_inputs: dict[str, Tensor], actions: Tensor):
        required={"nodes","image","scalars"}; missing=required-set(actor_inputs)
        if "edges" not in actor_inputs and "edge_index" not in actor_inputs: missing.add("edges/edge_index")
        if missing: raise KeyError(f"missing actor inputs: {sorted(missing)}")
        args={"nodes":actor_inputs["nodes"],"edges":actor_inputs.get("edges",actor_inputs.get("edge_index")),"image":actor_inputs["image"],"scalars":actor_inputs["scalars"],"actions":actions.to(self.device),"action_mask":actor_inputs.get("action_mask"),"node_mask":actor_inputs.get("node_mask")}
        args={k:(v.to(self.device) if isinstance(v,Tensor) else v) for k,v in args.items()}
        return self.policy.actor(stage).evaluate_actions(**args)
    def update(self, new_log_probs=None, old_log_probs=None, advantages=None, values=None, returns=None, *, batch: RolloutBatch|None=None, stage=None, actor_inputs=None, global_state=None, entropy=None, epochs=1):
        if epochs < 1: raise ValueError("epochs must be positive")
        mask = None
        if batch is not None:
            old_log_probs, advantages, returns = batch.log_probs, batch.advantages, batch.returns
            mask = batch.masks.to(self.device)
            if actor_inputs is not None and stage is None: raise ValueError("stage is required with actor_inputs")
        if epochs > 1 and (batch is None or actor_inputs is None or global_state is None):
            raise ValueError("multiple epochs require batch, actor_inputs, and global_state to recompute the policy")
        if any(x is None for x in (old_log_probs, advantages, returns)): raise ValueError("old log probabilities, advantages, and returns are required")
        old_log_probs,advantages,returns=[x.detach().to(self.device) for x in (old_log_probs,advantages,returns)]
        metrics={}
        for _ in range(epochs):
            if batch is not None and actor_inputs is not None:
                new_log_probs, entropy = self.evaluate_actor(stage, actor_inputs, batch.actions)
            if global_state is not None:
                values = self.policy.value(global_state.to(self.device))
            if new_log_probs is None or values is None: raise ValueError("actor log probabilities and critic values are required")
            new_log_probs, values = new_log_probs.to(self.device), values.to(self.device)
            entropy_value=torch.zeros((),device=self.device) if entropy is None else masked_mean(entropy.to(self.device), mask)
            actor_loss=ppo_loss(new_log_probs,old_log_probs,advantages,self.clip,mask); critic_loss=value_loss(values,returns,mask=mask)
            loss=actor_loss+self.value_coef*critic_loss-self.entropy_coef*entropy_value
            self.optimizer.zero_grad(set_to_none=True)
            grad_norm = torch.zeros((), device=self.device)
            if loss.requires_grad:
                loss.backward(); grad_norm = nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm); self.optimizer.step()
            with torch.no_grad():
                log_ratio = new_log_probs - old_log_probs
                ratio = log_ratio.exp()
                approx_kl = masked_mean((ratio - 1) - log_ratio, mask)
                clip_fraction = masked_mean(((ratio - 1).abs() > self.clip).float(), mask)
                valid = torch.ones_like(returns, dtype=torch.bool) if mask is None else mask.bool().expand_as(returns)
                target, residual = returns[valid], (returns - values)[valid]
                variance = target.var(unbiased=False) if target.numel() else torch.tensor(0., device=self.device)
                explained = 1 - residual.var(unbiased=False) / variance if target.numel() > 1 and variance > 0 else None
            metrics={"loss":float(loss.detach()),"actor_loss":float(actor_loss.detach()),"value_loss":float(critic_loss.detach()),"entropy":float(entropy_value.detach()),"approx_kl":float(approx_kl),"clip_fraction":float(clip_fraction),"gradient_norm":float(grad_norm),"learning_rate":float(self.optimizer.param_groups[0]["lr"])}
            if explained is not None: metrics["explained_variance"] = float(explained)
        return metrics
