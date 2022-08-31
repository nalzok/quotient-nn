.PHONY: main

main:
	pipenv run python3 \
		-m experiment.main \
		--batch_size 512 \
		--epochs 32 \
		--learning_rate 1e-2 \
		--scaling_epochs 512 \
		--scaling_rate 1e-5
