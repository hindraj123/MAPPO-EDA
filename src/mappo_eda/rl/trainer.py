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
        if batch is not None:
            old_log_probs, advantages, returns = batch.log_probs, batch.advantages, batch.returns
            if actor_inputs is not None:
                if stage is None: raise ValueError("stage is required with actor_inputs")
                new_log_probs, entropy = self.evaluate_actor(stage, actor_inputs, batch.actions)
            if new_log_probs is None: raise ValueError("new_log_probs or actor_inputs is required")
            values = self.policy.value(global_state.to(self.device)) if global_state is not None else (batch.values if values is None else values)
        if any(x is None for x in (new_log_probs,old_log_probs,advantages,values,returns)): raise ValueError("all PPO tensors are required")
        new_log_probs,old_log_probs,advantages,values,returns=[x.to(self.device) for x in (new_log_probs,old_log_probs,advantages,values,returns)]
        entropy_value=torch.zeros((),device=self.device) if entropy is None else masked_mean(entropy.to(self.device))
        metrics={}
        for _ in range(epochs):
            actor_loss=ppo_loss(new_log_probs,old_log_probs,advantages,self.clip); critic_loss=value_loss(values,returns)
            loss=actor_loss+self.value_coef*critic_loss-self.entropy_coef*entropy_value
            self.optimizer.zero_grad(set_to_none=True)
            if loss.requires_grad:
                loss.backward(); nn.utils.clip_grad_norm_(self.policy.parameters(),self.max_grad_norm); self.optimizer.step()
            metrics={"loss":float(loss.detach()),"actor_loss":float(actor_loss.detach()),"value_loss":float(critic_loss.detach()),"entropy":float(entropy_value.detach())}
        return metrics
