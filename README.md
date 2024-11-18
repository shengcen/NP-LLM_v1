
## Benchmarks:
$$ python teal.py --obj throughput --topo UsCarrier.json --tm-model toy --epochs 1 --admm-steps 50


## build the env first

$$ conda activate molca

$$ cd cs/Molca/

### Behavioral Cloning to generate the initial solution for the test set (i.e. the version of IEE CommMag)
# Step 1: training
$$ python stage2.py --root 'data/my_dataset/' --devices '1' --filename "ft_pubchem324k" --opt_model 'llama2_13b_hf' --max_epochs 800 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --batch_size 8 --inference_batch_size 8





# Step 2: inference
$$ python stage2.py --root 'data/my_dataset/' --devices '1' --filename "ft_pubchem324k" --stage2_path "all_checkpoints/ft_pubchem324k/epoch=1599.ckpt" --opt_model 'llama2_13b_hf' --max_epochs 400 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --batch_size 1 --inference_batch_size 1 --init_checkpoint "all_checkpoints/ft_pubchem324k/epoch=1599.ckpt"



