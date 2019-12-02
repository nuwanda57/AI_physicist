import AI_physicist.theory_learning.options as options
import datetime


class PendulumOptions(options.Options):
    def __init__(self):
        super().__init__()
        self.num_layers = 5
        self.pred_nets_neurons = 160
        self.domain_net_neurons = 16
        self.reg_domain_amp = 1e-7
        self.show_3D_plot = False
        self.pred_nets_activation = "tanh"
        self.csv_filename_list = [
            'mysteryt_px_10', 'mysteryt_px_11', 'mysteryt_px_12', 'mysteryt_px_13',
            'mysteryt_px_20', 'mysteryt_px_21', 'mysteryt_px_23']
        self.is_simplify_model = False
        self.is_simplify_domain = False
        self.add_theory_loss_threshold = 1e-4
        self.num_theories_init = 2
        self.add_theory_limit = 2
        self.is_xv = True
        self.record_mode = 1
        #     reg_smooth = (0.1, 2, 10, 1e-2, 1)
        if self.is_xv:
            self.csv_filename_list = ['mysteryt_pxv_10', 'mysteryt_pxv_11', 'mysteryt_pxv_12', 'mysteryt_pxv_13',
                                 'mysteryt_pxv_20', 'mysteryt_pxv_21', 'mysteryt_pxv_23']
            self.num_output_dims = 4
            self.num_input_steps = 1
            self.add_theory_loss_threshold = 3e-4
        for key in self.csv_filename_list:
            self.big_domain_dict[key] = [0, 1]
        self.is_pendulum = True
