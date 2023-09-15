import mlflow
import os
import hydra
from omegaconf import DictConfig, OmegaConf


# This automatically reads in the configuration
@hydra.main(config_name='config')
def go(config: DictConfig):

    # Setup the wandb experiment. All runs will be grouped under this name
    os.environ["WANDB_PROJECT"] = config["main"]["project_name"]
    os.environ["WANDB_RUN_GROUP"] = config["main"]["experiment_name"]

    # You can get the path at the root of the MLflow project with this:
    root_path = hydra.utils.get_original_cwd()

    # Check which steps we need to execute
    if isinstance(config["main"]["execute_steps"], str):
        # This was passed on the command line as a comma-separated list of steps
        steps_to_execute = config["main"]["execute_steps"].split(",")
    else:
        assert isinstance(config["main"]["execute_steps"], list)
        steps_to_execute = config["main"]["execute_steps"]

    # Download step
    if "download" in steps_to_execute:

        _ = mlflow.run(
            os.path.join(root_path, "download"),
            "main",
            parameters={
                "file_url": config["data"]["file_url"],
                "artifact_name": "raw_data.parquet",
                "artifact_type": "raw_data",
                "artifact_description": "Data as downloaded"
            },
        )

    if "preprocess" in steps_to_execute:

        ## YOUR CODE HERE: call the preprocess step
        _ = mlflow.run(
            os.path.join(root_path, "preprocess"),
            parameters=dict(
                input_artifact="raw_data.parquet:latest",
                artifact_name="preprocessed_data.csv",
                artifact_type="clean_data",
                artifact_description="Preprocess data"
            )
        )

    if "check_data" in steps_to_execute:

        ## YOUR CODE HERE: call the check_data step
        _ = mlflow.run(
            os.path.join(root_path, 'check_data'),
            parameters=dict(
                reference_artifact=config['data']['reference_dataset'],
                # The sample_artifact should be a different data set
                # Using the same one here...
                sample_artifact=config['data']['reference_dataset'],
                ks_alpha=config['data']['ks_alpha'],
            )
        )

    if "segregate" in steps_to_execute:

        ## YOUR CODE HERE: call the segregate step
        mlflow.run(
            os.path.join(root_path, 'segregate'),
            parameters=dict(
                input_artifact="preprocessed_data.csv:latest",
                artifact_root="data",
                artifact_type="split_data",
                test_size=config["data"]["test_size"],
                random_state=config["main"]["random_seed"],
                stratify=config["data"]["stratify"]
            )
        )

    if "random_forest" in steps_to_execute:

        # Serialize decision tree configuration
        model_config = os.path.abspath("random_forest_config.yml")
        # Compute model_config relpath from random forest path
        # - mlflow parameters must have at most 250 characters
        random_forest_path = os.path.join(root_path, "random_forest")
        model_config_relpath_from_model_path = os.path.relpath(
            model_config, start=random_forest_path)

        with open(model_config, "w+") as fp:
            fp.write(OmegaConf.to_yaml(config["random_forest_pipeline"]))

        ## YOUR CODE HERE: call the random_forest step
        mlflow.run(
            random_forest_path,
            parameters=dict(
                train_data="exercise_14/data_train.csv:latest",
                model_config=model_config_relpath_from_model_path,
                export_artifact=(
                    config['random_forest_pipeline']['export_artifact']),
                random_seed=(
                    config['random_forest_pipeline']['random_forest']
                    ['random_state']),
                val_size=config['data']['val_size'],
                stratify=config['data']['stratify']
            )
        )

    if "evaluate" in steps_to_execute:

        ## YOUR CODE HERE: call the evaluate step
        export_artifact = config['random_forest_pipeline']['export_artifact']
        mlflow.run(
            os.path.join(root_path, "evaluate"),
            parameters=dict(
                test_data="exercise_14/data_test.csv:latest",
                model_export=f"{export_artifact}:latest"
            )
        )


if __name__ == "__main__":
    go()
