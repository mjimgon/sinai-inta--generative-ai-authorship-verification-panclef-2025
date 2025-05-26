# sinai-inta--generative-ai-authorship-verification-panclef-2025
sinai-inta on Task: generative-ai-authorship-verification-panclef-2025

1º Unzip the model 
modelo_tfid_todo.pkl

To run the model, please use: 

python ./modelo_final.py [input_data] [output_directory] 


tira-run --input-dataset generative-ai-authorship-verification-panclef-2025/pan25-generative-ai-detection-val   --image submission:latest   --command 'python ./modelo_final.py'


## Submission to TIRA
1. Check if the code works:
  ```bash
  tira-cli code-submission --dry-run --path ./ --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training
  ```

2. If that ran successfully, you can omit the `-dry-run` argument to submit:
  ```bash
  tira-cli code-submission --path ./ --task generative-ai-authorship-verification-panclef-2025 --dataset pan25-generative-ai-detection-smoke-test-20250428-training
  ```