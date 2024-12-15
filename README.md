## Towards Real-Time Symbolic Music Generation on Constrained Devices
We experiment with a technique called guidance (Subramaniam et al 2024) to improve symbolic music generation on fast recurrent models.

## Running Guidance
To run baseline models, use the following command
```
python -m music-gen.music_gen --exp_name [EXP_NAME] --student_model Music-LSTM --batch_size 4 --task maestro --lr 1.0 --num_epochs 100
```

To run guidance, use the following command
```
python -m music-gen.music_gen --exp_name [EXP_NAME] --batch_size 4 --student_model Music-LSTM --task maestro --lr 1.0 --rep_sim --repdist CKA --target_model music-trans --untrained --num_epochs 100
```

## Inference Demonstration
We share a google colab that will download all of the trained models for all the LSTMs both guided and unguided.
It is available here: https://colab.research.google.com/drive/1VJX512hYzBBxrtaiQCoLNtET9yzmNY5Y?usp=sharing

Feel free to run the code or scroll through and listen to the samples. If you decide to run the code, please use CUDA and note that inference works well with teacher-forcing for the LSTMs in which the hidden state is recalculated from the whole context at every time step. This slows down inference which is why we encourage the use of CUDA here. We aim to remove the reliance on teacher-forcing in future, which will hopefully demonstrate the improved inference that we show in this demo for the Bach Chorales models while also having the low latency we observe when not using teacher-forcing.

## Inference Time
We calculate the inference time figure using music_gen/inference_time.ipynb on a macbook pro.
