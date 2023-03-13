from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--train_valiSize', type=int, default=336, help='number of train and validation images')
        parser.add_argument('--testSize', type=int, default=64, help='number of test images')
        parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
        parser.add_argument('--netGpath', type=str, required=True, help='path of trained netG model')
        parser.add_argument('--visualize', action='store_true', help='output images for visualization')
        parser.add_argument('--smoothing', type=str, default='none', help='type of smoothing, [none, RSE2E, SMUGv0, SMUG]')
        parser.add_argument('--num_sample', type=int, default=10, help='number of samples in smoothing')
        parser.add_argument('--smoothing_epsilon', type=float, default=0.01, help='epsilon of gaussian noise')
        parser.add_argument('--pgd_steps', type=int, default=10, help='number of steps in PGD attack')
        parser.add_argument('--acceleration', type=float, default=4, help='acceleration factor of mask')

        self.isTrain = False
        return parser
