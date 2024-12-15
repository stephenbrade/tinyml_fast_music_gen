# Improved Symbolic Music Generation

## Running Guidance
To run baseline models, use the following command
```
python -m music-gen.music_gen --exp_name [EXP_NAME] --student_model Music-LSTM --batch_size 4 --task maestro --lr 1.0
```

To run guidance, use the following command
```
python -m music-gen.music_gen --exp_name [EXP_NAME] --batch_size 4 --student_model Music-LSTM --task maestro --lr 1.0 --rep_sim --repdist CKA --target_model music-trans --untrained
```

