### AI OPs Assignment 1 Q4 Submission
- By Madhav AK and Krish Dange
- DA24B012 and DA24B011
- Partner A = Madhav AK
- Partner B = Krish Dange

### Process:

- This Repo was created by Partner A (Madhav) with all the necessary environment.yml, requirement.txt, and all necessary dvc and training code related files and pushed to github.
- BackBlaze was used as the Remote DVC storage
- Partner A ran the code on mlflow and took note of the MLFlow readings (which can be found in the parterA_proof/ folder
- Partner B pulled the repo, and with no other instructions from parter A, used the permitted commands to independently reproduce the results. (The readings can be found in the parterB_proof/ folder)
- Partner B then checked that all the results were identical and the environment was indeed reproduced correctly.
- Everything related to this question were done on this repo. This Repo's history can be verified to see if all processses were followed.

### Note on Video Recordings:

- Each partner's individual video recordings can be found in their original submission repository.

### Instructions to the run the code:

- The file data_creation.py can be run to simply download the iris database and save it as a csv
- python data_creation.py
- Launch an MLFlow server on your terminal using:
- mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000 --allowed-hosts "*" --cors-allowed-origins "http://localhost:5000, http://127.0.0.1:5000"
- The file train_model.py can now be run to connect to the server, train, run and register the model
- python train_model.py
