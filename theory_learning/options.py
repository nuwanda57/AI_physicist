from AI_physicist.theory_learning.util_theory import get_mystery
import datetime


class Options(object):
    def __init__(self):
        ########################################
        # Setting up path to dataset files. The data files are CSV_DIRNAME + env_name + ".csv",
        # where env_name are elements in the csv_filename_list. For each csv file, the first
        # (num_output_dims * num_input_steps) columns are the past (E.g., if num_output_dims=2, then
        # they are arranged as (x_{t-num_input_steps}, y_{t-num_input_steps}, ... x_{t-1}, y_{t-1}) ).
        # The next num_output_dims columns are the target prediction for the future.
        # If is_classified = True, the last column in the csv should provide the true_domain id for evaluating
        # whether the domain prediction is correct (not used for training)
        ########################################
        self.is_classified = True  # If True, the last column in the csv file should provide the true_domain id for
        # evaluation.
        self.csv_filename_list = get_mystery(
            50000, range(4, 7), range(1, 6), self.is_classified) + get_mystery(
            50000, [20], range(1, 6), self.is_classified) + get_mystery(50000, range(7, 11), range(1, 6),
                                                                    self.is_classified)
        self.num_output_dims = 2  # It sets the dimension of output
        self.num_input_steps = 2  # It sets the number of steps for the input
        self.exp_mode = "continuous"  # Choose from "continuous" (full AI Physicist), "newb" (newborn) and "base" (
        # baseline)
        self.forward_steps = 1  # Number of forward steps to predict
        self.data_format = "states"  # Choose from "states" or "images"
        self.pred_nets_activation = "linear"  # Activation for the prediction function f. Choose from "linear",
        # "leakyRelu"
        self.num_layers = 3  # Number of layers for the prediction function f.

        self.num_theories_init = 4  # Number of theories to start with.
        self.add_theory_loss_threshold = 2e-6  # MSE threshold for individual data points above which to add a new
        # theory to fit.
        self.add_theory_criteria = ("loss_with_domain",
                               0)  # Criteria and threshold of loss increase to determine whether to accept adding the theory.
        self.add_theory_quota = 1  # maximum number of theories to add at each phase.
        self.add_theory_limit = None  # maximum allowed number of theories. If None, do not set limit.

        self.is_Lagrangian = False  # If True, learn the Lagrangian. If False, learn Equation of Motion (EOM).
        self.load_previous = True  # Whether to load previously trained instances on

        # Other settings:
        self.exp_id = "exp1.0"
        self.env_source = "file"
        self.pred_nets_neurons = 8
        self.domain_net_neurons = 8
        self.domain_pred_mode = "onehot"
        self.mse_amp = 1e-7
        self.scheduler_settings = ("ReduceLROnPlateau", 40, 0.1)  # Settings for the learning rate scheduler
        # scheduler_settings = ("LambdaLR", "exp", 2, False)    # Settings for the learning rate scheduler
        self.simplify_criteria = ("DLs", 0, 3,
                             "relative")  # The (criteria type, threshold, patience, compare_mode) upon which not satisfied, we break the current simplification and continue to the next layer/model
        self.optim_type = ("adam", 5e-3)
        self.optim_domain_type = ("adam", 1e-3)
        self.optim_autoencoder_type = ("adam", 1e-5, 1e-1)  # optim_type, lr, loss_scale
        self.reg_mode = "L1"
        self.reg_amp = 1e-8
        self.reg_smooth = None
        self.reg_domain_mode = "L1"
        self.reg_domain_amp = 1e-5
        self.batch_size = 10000
        self.loss_core = "DLs"
        self.loss_order = -1
        self.loss_decay_scale = None
        self.is_mse_decay = False
        self.num_examples = 20000
        self.epochs = 10000
        self.iter_to_saturation = int(self.epochs / 2)
        self.MDL_mode = "both"
        self.date_time = "{0}-{1}".format(datetime.datetime.now().month, datetime.datetime.now().day)
        self.seed = 0
        self.array_id = "0"

        self.loss_balance_model_influence = False
        self.loss_success_threshold = 1e-4  # MSE level you regard as success
        self.theory_add_mse_threshold = 0.05  # MSE level below which you will add to the theory hub
        self.theory_remove_fraction_threshold = 0.005  # Fraction threshold below which you will remove a theory after each stage of training.
        self.matching_numerical_tolerance = 2e-4  # The tolerance below which you regard the numerical coefficient matches.
        self.matching_snapped_tolerance = 1e-9  # The tolerance below which you regard the snapped coefficient matches.
        self.max_trial_times = 1  # Maximum number of trial times before going on to next target (DEFAULT=1)
        self.is_simplify_model = True  # Whether to perform simplification of theory models
        self.is_simplify_domain = False  # Whether to perform simplification of theory domains
        self.record_mode = 2  # Record data mode. Choose from 0 (minimal recording), 1, 2 (record everything)
        self.show_3D_plot = False
        self.show_vs = False
        self.big_domain_dict = [
                (key, [1, 2]) for key in get_mystery([
                    20000, 30000, 40000, 50000], range(4, 7), range(11))] + [(key, [1, 2]) for key in get_mystery(
                        [40000, 50000], [20], range(11))] + [(key, [1, 2, 3]) for key in get_mystery(
                            [20000, 30000, 40000, 50000], range(7, 10), range(11))] + [
                                                    (key, [1, 2, 3, 4]) for key in get_mystery(
                                                            [20000, 30000, 40000, 50000], [10], range(11))]
        self.big_domain_dict = {key: item for key, item in self.big_domain_dict}

        # Settings for data_format = "images":
        if self.data_format == "images":
            self.batch_size = 100
            self.epochs = 10000
            self.loss_core = "mse"
            self.add_theory_quota = 0
            self.is_simplify_model = False
            self.is_simplify_domain = False

        # Settings for Lagrangian:
        if self.is_Lagrangian:
            self.num_input_steps = 3
            self.is_simplify_model = False

        self.is_pendulum = False
