"""
BERT模型压缩项目 - 源代码包
包含模型加载、量化、剪枝、训练和评估等模块
"""

from . import model
from . import quantization
from . import pruning
from . import train
from . import evaluate
from . import utils

__version__ = '1.0.0'
__author__ = 'ML3 Project'

# 便捷导入
from .model import create_bert_model, BERTModelManager
from .quantization import quantize_bert_model, BERTQuantizer
from .pruning import prune_bert_model, BERTPruner
from .train import finetune_model, progressive_finetune
from .evaluate import ModelEvaluator
from .utils import set_seed, get_model_size, count_parameters

__all__ = [
    'model',
    'quantization', 
    'pruning',
    'train',
    'evaluate',
    'utils',
    'create_bert_model',
    'BERTModelManager',
    'quantize_bert_model',
    'BERTQuantizer',
    'prune_bert_model',
    'BERTPruner',
    'finetune_model',
    'progressive_finetune',
    'ModelEvaluator',
    'set_seed',
    'get_model_size',
    'count_parameters'
]
