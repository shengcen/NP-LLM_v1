
## Benchmarks:
In blip2_llama.py, set is_teal = True, use_prompt = False     (for one-pass training/inference, and the other files are followed as belows); or 
$$ python teal.py --obj throughput --topo UsCarrier.json --tm-model toy --epochs 1 --admm-steps 50   (for RL searching in the test set)


## build the env first

$$ conda activate molca

$$ cd cs/Molca/

### Behavioral Cloning to generate the initial solution for the test set (i.e. the version of IEE CommMag)
In blip2_llama.py, set is_teal = False, use_prompt = True (for the version of IEE CommMag) or False (for the new version)


# Step 1: training
In stage2.py, set "IS_TRAINING"=True;
In data/my_dataset/my_dataset_writer.py, set is_test=False; ; run python my_dataset_writer.py


$$ python stage2.py --root 'data/my_dataset/' --devices '1' --filename "ft_pubchem324k" --opt_model 'llama2_13b_hf' --max_epochs 800 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --batch_size 8 --inference_batch_size 8





# Step 2: inference
In stage2.py, set "IS_TRAINING"=False;
In data/my_dataset/my_dataset_writer.py, set is_test=True; edit the traffic, network information, and task type to the test set in "if is_test:"; run python my_dataset_writer.py
In blip2_llama.py, edit the traffic, network information, and task type to the test set in the outmost space;


$$ python stage2.py --root 'data/my_dataset/' --devices '1' --filename "ft_pubchem324k" --stage2_path "all_checkpoints/ft_pubchem324k/epoch=1599.ckpt" --opt_model 'llama2_13b_hf' --max_epochs 400 --mode ft --prompt '[START_I_SMILES]{}[END_I_SMILES]. ' --tune_gnn --llm_tune lora --batch_size 1 --inference_batch_size 1 --init_checkpoint "all_checkpoints/ft_pubchem324k/epoch=1599.ckpt"



