from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--trainSize', type=int, default=304, help='train data size')
        parser.add_argument('--valiSize', type=int, default=32, help='validation data size')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--epoch', type=int, default=50, help='the number of epoches')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        parser.add_argument('--name', type=str, default='experiment_name', help='name of the experiment. It decides where to store samples and models')
        parser.add_argument('--lr', type=float, default=1e-4, help='initial learning rate for adam')
        parser.add_argument('--pgd_steps', type=int, default=10, help='number of steps of PGD attack')
        parser.add_argument('--pgd_alpha', type=float, default=0.5/255, help='alpha in PGD attack')
        parser.add_argument('--pgd_epsilon', type=float, default=1/255, help='epsilon in PGD attack')
        parser.add_argument('--pgd_type', type=str, default='linfty', help='l2/linfty PGD attack')
        parser.add_argument('--smoothing', type=str, default='none', help='type of smoothing, [none, RSE2E, SMUGv0, SMUG]')
        parser.add_argument('--num_sample', type=int, default=10, help='number of samples in smoothing')
        parser.add_argument('--smoothing_epsilon', type=float, default=0.01, help='epsilon of gaussian noise')
        parser.add_argument('--LossLambda', type=float, default=1.0, help='Lambda in loss function')

        self.isTrain = True
        return parser
