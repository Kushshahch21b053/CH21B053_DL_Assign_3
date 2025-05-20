# DL Assignment 3

### Github Link

https://github.com/Kushshahch21b053/CH21B053_DL_Assign_3

### Wandb Report Link

https://wandb.ai/ch21b053-indian-institute-of-technology-madras/dakshina-seq2seq/reports/CH21B053-A3-Report--VmlldzoxMjgyOTUzMg?accessToken=otr8p6zqmx40u9y48nv46y53nwmuk1j0bpt9zgl5inb5g4y3sqce79ihs9kz226h

### Code Organisation

- `config.py`
- `data_utils.py`
- `evaluate.py`
- `generate_predictions.py`
- `model_attention.py`
- `model_vanilla.py`
- `plots_attention.py`
- `sample_connectivity.py`
- `sample_vanilla.py`
- `test_attention.py`
- `test_vanilla.py`
- `train_attention.py`
- `train_vanilla.py`

### How to run code

- Firstly, if needed do:
```
pip install -r requirements.txt
```

Place the dakshina_dataset_v1.0 folder in the root directory. Update the path in config.py if required.

**Question 1-3:**

- To do a local vanilla single run:
```
python train_vanilla.py
--embed_dim <EMBED_DIM>
--hidden_dim <HIDDEN_DIM>
--enc_layers <ENC_LAYERS>
--dec_layers <DEC_LAYERS>
--cell_type <SimpleRNN|GRU|LSTM>
--dropout <DROPOUT>
--batch_size <BATCH_SIZE>
--epochs <EPOCHS>
```

- To perform the sweep for the vanilla model, do:
```
python train_vanilla.py
--sweep
--project <PROJECT_NAME>
--entity <ENTITY_NAME>
--count <NUM_RUNS>
```

**Question 4:**

- To retrain the vanilla model with the best hyperparameters and evaluate on the test set, do:
```
python test_vanilla.py
```

- To see sample predictions made by this model, as well as the folder containing all the predictions on the test set, do:
```
python sample_vanilla.py
```

**Question 5:**

- To do a local attention single run, do:
```
python train_attention.py
--embed_dim <EMBED_DIM>
--hidden_dim <HIDDEN_DIM>
--dropout <DROPOUT>
--batch_size <BATCH_SIZE>
--epochs <EPOCHS>

text
```

- To perform the sweep for the attention model, do:
```
python train_attention.py
--sweep
--project <PROJECT_NAME>
--entity <ENTITY_NAME>
--count <NUM_RUNS>
```

- To retrain the attention model with the best hyperparameters and evaluate on the test set, as well as create the folder to save all predictions, do:
```
python retrain_attention.py
```

- To see sample predictions made by this model, as well as the comparisons between vanilla and attention models, do:
```
python sample_attention.py
```

**Question 6:**

- To show the connectivity visualisation, do:
```
python sample_connectivity.py
--num_examples <N>
--hidden_dim <HIDDEN_DIM>
--model_path models/best_seq2seq_attention.h5
```