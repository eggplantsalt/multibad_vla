# 验证记录

## Smoke Test

```bash
python -c "from prismatic.vla.research.compositional import CueManager, ViewBuilder, ThreeViewBatchTransform; cm=CueManager(); vb=ViewBuilder(cm); t=ThreeViewBatchTransform(lambda s: s, vb); out=t({'x': 1}); assert out.clean['x']==1 and out.incomplete['x']==1 and out.full['x']==1"
```

- 预期结果：无异常退出，且三视图结构可用。

## Three-View Dry Run

```bash
python vla-scripts/finetune_with_task_three_view.py \
	--enable_three_view True \
	--vla_path openvla/openvla-7b \
	--data_root_dir /PATH/TO/RLDS \
	--dataset_name libero_spatial_no_noops \
	--run_root_dir ./runs \
	--batch_size 1 \
	--max_steps 2 \
	--save_freq 1 \
	--save_latest_checkpoint_only True
```

- 预期结果：日志中包含三路 loss，且 checkpoint 保存流程不报错。
