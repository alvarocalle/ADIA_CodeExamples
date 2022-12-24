This project uses: packrat to manage the environment and isolate code

Step 1: create the project with packrat enabled (either in Terminal or RStudio -> details in notes.txt)
Step 2: list in libs.txt all the needed libraries in csv format (sep=",")
Step 3: run set_packrat_env.R. This script will install all the needed libraries in the environment
Step 4: run model_train.R to train the ML model and save results as rds files
Step 5: create scoring script model_scoring.R and API with plumber
Step 6: run api.R
Step 7: use curl from the terminal and feed in a JSON file with feature values
        curl -X POST "http://127.0.0.1:8000/score" -H  "Content-Typ: application/json" --data @data.json


