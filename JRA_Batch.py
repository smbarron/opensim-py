"""
Joint reactions analysis batch processing script.

This script runs the joint reactions analysis tool on a specified list of subjects, using a combinaiton of measured and predicted muscle activations.
It is set up to be dependent upon the outputs of the static optimization tool, which should be run first. It is additionally dependent upon the osim_emg
package, available at https://github.com/smbarron/osim-emg.

Usage:
    python JRA_Batch.py <tear_subjects_file> <no_tear_subjects_file> <split_emg_filepath> <not_split_emg_filepath> <root_dir> <setup_template>

Arguments:
    tear_subjects_file: Path to a text file containing a list of subject IDs with tears. Each subject ID should be on a separate line.
    no_tear_subjects_file: Path to a text file containing a list of subject IDs without tears. Each subject ID should be on a separate line.
    split_emg_filepath: Path to the split EMG data file. This should be a .pkl file containing a dictionary of the form {(subject, trial): List[emg_data]},
        where emg_data is a pandas dataframe containing the EMG data for a single task repetition within a trial. The EMG data should be pre-filtered and
        normalized to maximum contraction values.
    not_split_emg_filepath: Path to the non-split EMG data file. This should be a .pkl file containing a dictionary of the form {(subject, trial): emg_data},
        where emg_data is a pandas dataframe containing the EMG data for whole trial. The EMG data should be pre-filtered and
        normalized to maximum contraction values.
    root_dir: Path to the directory containing the data. The directory structure should be as follows:
        root_dir
        ├── subject_id_1
        │   ├── subject_id_1_clamped.osim or subject_id_1_nosupra.osim (This is an already scaled model with or without the supraspinatus.)
        │   ├── SO_Results
        │   │   ├── trial_1_SO_StatesReporter_states.sto
        │   │   ├── trial_1_SO_StaticOptimization_activation.sto
        │   │   ├── trial_2_SO_StatesReporter_states.sto
        │   │   ├── trial_2_SO_StaticOptimization_activation.sto
        │   │   └── ...
        ├── subject_id_2
        │   ├── subject_id_2_clamped.osim or subject_id_2_nosupra.osim
        │   ├── SO_Results
        │   │   ├── trial_1_SO_StatesReporter_states.sto
        │   │   ├── trial_1_SO_StaticOptimization_activation.sto
        │   │   ├── trial_2_SO_StatesReporter_states.sto
        │   │   ├── trial_2_SO_StaticOptimization_activation.sto
        │   │   └── ...
        └── ...
    setup_template: Path to the setup template file (JRA_Setup.xml).

Example Usage:
    'python JRA_batch.py
        osim_emg/tear.txt
        osim_emg/notear.txt
        osim_emg/splitnorm_EMG_cleaned.pkl
        osim_emg/normalized_emg_not_split.pkl
        test_EMG_package
        JRA_setup.xml'
"""

import logging
import os
import sys
from typing import List, Optional, Tuple

import opensim as osim
from osim_emg.activations_to_states import combine_files
from osim_emg.merge_measured_with_predicted import save_combined_activations

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def create_jra_setup(
    setup_template: str,
    model_file: str,
    states_file: str,
    split_emg_filepath: str,
    not_split_emg_filepath: str,
    initial_time: float,
    final_time: float,
    results_dir: str,
    trial_name: float,
    cohort: Optional[bool],
) -> str:
    """Find and replace in the setup .xml template with trial info and save it as a new file."""
    with open(setup_template, "r") as f:
        content = f.read()

    combined_states = combine_files(states_file)
    activation_file = states_file.replace("StatesReporter_states", "StaticOptimization_activation")
    combined_activations = save_combined_activations(activation_file, split_emg_filepath, not_split_emg_filepath, cohort)

    content = content.replace("MODEL_FILE", model_file)
    content = content.replace("COMBINED_STATES_FILE", combined_states)
    content = content.replace("COMBINED_ACTIVATION_FILE", combined_activations)
    content = content.replace("INITIAL_TIME", str(initial_time))
    content = content.replace("FINAL_TIME", str(final_time))
    content = content.replace("RESULTS_DIRECTORY", results_dir)
    content = content.replace("TRIAL_NAME", trial_name)

    setup_file = f"{trial_name}_Setup.xml"
    setup_path = os.path.join(results_dir, setup_file)

    with open(setup_path, "w") as f:
        f.write(content)
        f.close()

    return rf"{setup_path}"


def get_subject_info(root: str, subject_id: str, tear_subjects: List[str], no_tear_subjects: List[str]) -> Tuple[str, str, str, bool]:
    """Get the model file, SO results directory, JRA results directory, and cohort for a given subject."""
    if subject_id in tear_subjects:
        model_file = rf"{root}/{subject_id}/{subject_id}_nosupra.osim"
        so_dir = model_file.replace(f"{subject_id}_nosupra.osim", "SO_Results")
        results_dir = model_file.replace(f"{subject_id}_nosupra.osim", "JRA_Results")
        cohort = True

    elif subject_id in no_tear_subjects:
        model_file = rf"{root}/{subject_id}/{subject_id}_clamped.osim"
        so_dir = model_file.replace(f"{subject_id}_clamped.osim", "SO_Results")
        results_dir = model_file.replace(f"{subject_id}_clamped.osim", "JRA_Results")
        cohort = False
    model_file_path = os.path.abspath(model_file)
    so_dir_path = os.path.abspath(so_dir)
    results_dir_path = os.path.abspath(results_dir)

    return model_file_path, so_dir_path, results_dir_path, cohort


def prep_joint_reactions_analysis(
    setup_template: str,
    model_file: str,
    states_file: str,
    results_dir: str,
    split_emg_filepath: str,
    not_split_emg_filepath: str,
    trial_name: str,
    cohort: Optional[bool],
) -> str:
    """Get trial time info from states file and create a setup file for the JRA tool."""
    states_data = osim.Storage(states_file)
    initial_time = states_data.getFirstTime()
    final_time = states_data.getLastTime()

    setup_file = create_jra_setup(
        setup_template, model_file, states_file, split_emg_filepath, not_split_emg_filepath, initial_time, final_time, results_dir, trial_name, cohort
    )

    return setup_file


def run_joint_reactions_analysis(
    tear_subjects: List[str], no_tear_subjects: List[str], split_emg_filepath: str, not_split_emg_filepath: str, root_dir: str, setup_template: str
):
    """Run the joint reactions analysis tool on a list of subjects."""
    subject_list = tear_subjects + no_tear_subjects
    for subject_id in subject_list:
        cohort = None
        logging.info(f"Processing subject: {subject_id}")

        model_file, so_dir, results_dir, cohort = get_subject_info(root_dir, subject_id, tear_subjects, no_tear_subjects)

        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
            logging.info(f"Created results directory: {results_dir}")

        for _, _, filenames in os.walk(so_dir):
            for name in filenames:
                if "_states.sto" in name:
                    states_file = so_dir + "/" + name
                    sub_string = states_file.split("/")
                    trial = sub_string[-1].split("_SO")[0]
                    tool_name = trial + "_JRA"

                    logging.info(f"Trial: {trial} in progress...")

                    setup_file = prep_joint_reactions_analysis(
                        setup_template, model_file, states_file, results_dir, split_emg_filepath, not_split_emg_filepath, tool_name, cohort
                    )
                    model = osim.Model(model_file)
                    state = model.initSystem()  # noqa: F841
                    analyze_tool = osim.AnalyzeTool(setup_file, True)
                    analyze_tool.run()

                    logging.info(f"Trial: {trial} complete!")

        logging.info(f"Subject: {subject_id} complete!")
    logging.info("All subjects complete!")


if __name__ == "__main__":
    if len(sys.argv) != 7:
        logging.info(
            "Usage: python JRA_Batch.py <tear_subjects_file> <no_tear_subjects_file> <split_emg_filepath> <not_split_emg_filepath> <root_dir> <setup_template>"
        )
        sys.exit(1)

    tear_subjects_file = sys.argv[1]
    no_tear_subjects_file = sys.argv[2]
    split_emg_filepath = sys.argv[3]
    not_split_emg_filepath = sys.argv[4]
    root_dir = sys.argv[5]
    setup_template = sys.argv[6]

    with open(tear_subjects_file, "r") as file:
        tear_subjects = file.read().splitlines()

    with open(no_tear_subjects_file, "r") as file:
        no_tear_subjects = file.read().splitlines()

    run_joint_reactions_analysis(tear_subjects, no_tear_subjects, split_emg_filepath, not_split_emg_filepath, root_dir, setup_template)
