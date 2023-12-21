torchrun \
	--nnodes=1 \
	--nproc-per-node=2 \
	--rdzv-id=4242 \
	--rdzv-backend=c10d \
	--rdzv-endpoint="localhost:6137" \
	mrunner_run.py --ex mrunner_exps/blueprints/@-AA-BC_scaled.py
