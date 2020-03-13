import pandas as pd
import numpy as np
import os, random, math, glob
from fastai.vision import *
import fastai
#from sklearn.metrics import log_loss as skll
from datetime import datetime
import logging
from assets.models import pretrainedmodels
from assets.models import efficientnets

# We get to see the log output for our execution, so log away!
logging.basicConfig(level=logging.INFO)


def get_efficientnet(B="b1"):
    model = efficientnets.EfficientNet.from_name('efficientnet-'+B)
    model_name = 'efficientnet-'+B
    image_size = efficientnets.EfficientNet.get_image_size(model_name)

    model = efficientnets.EfficientNet.from_pretrained(model_name)

    FC={"b1":1280, "b3":1536}
    model.add_module('_fc',nn.Linear(FC[B], 54))
    return model

def get_srx50(pretrained=False,**kwargs):
    return pretrainedmodels.se_resnext50_32x4d(num_classes=1000,pretrained=None)

def open_croped_image1(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
    "Return `Image` object created from image in file `fn`."
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        try:
            #print(fn)
            x = PIL.Image.open(fn).convert(convert_mode)           
        except:
            print("\t\t",fn,"corrupt")
            x = PIL.Image.new('RGB', (512, 384)).convert(convert_mode)   
    if after_open: x = after_open(x)    
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(x)
vision.data.open_image = open_croped_image1

def open_croped_image_flipped(fn:PathOrStr, div:bool=True, convert_mode:str='RGB', cls:type=Image,
        after_open:Callable=None)->Image:
    "Return `Image` object created from image in file `fn`."
    
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning) # EXIF warning from TiffPlugin
        try:
            # print(fn)
            x = PIL.Image.open(fn).convert(convert_mode).transpose(PIL.Image.FLIP_LEFT_RIGHT)          
        except:
            print("\t\t",fn,"corrupt")
            x = PIL.Image.new('RGB', (512, 384)).convert(convert_mode)   
    if after_open: x = after_open(x)    
    x = pil2tensor(x,np.float32)
    if div: x.div_(255)
    return cls(x)

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

seed_everything(0)


# class AggLogLoss(Callback):
#     "Wrap a `func` in a callback for metrics computation."
#     def __init__(self, func=0, f2=0, **kwargs):
#         self.v = 0

#     def on_epoch_begin(self, **kwargs):
#         "Set the inner value to 0."
#         self.val, self.count = 0.,0
#         self.log_losses = []

#     def on_batch_end(self, last_output, last_target, **kwargs):
#         "Update metric computation with `last_output` and `last_target`."
#         if not is_listy(last_target): last_target=[last_target]
#         lls = skll(last_target[0], last_output)
#         self.log_losses.append((last_target[0].size(0), lls))
#         self.count += 1

#     def on_epoch_end(self, last_metrics, **kwargs):
#         "Set the final result in `last_metrics`."
#         return add_metrics(last_metrics, sum([l[1]/l[0] for l in self.log_losses])/self.count)
    

def model_f(pretrained=True,**kwargs):
    return pretrainedmodels.se_resnext50_32x4d(num_classes=1000,pretrained='imagenet')

class Head(nn.Module):
    def __init__(self, f_in, num_classes=1000, p=0.0):
        super(Head, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(p)
        self.fc = nn.Linear(f_in, num_classes)

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(self.dropout(x))
        return x
    
class AccumulateOptimWrapper(OptimWrapper):
    def step(self):           pass
    def zero_grad(self):      pass
    def real_step(self):      super().step()
    def real_zero_grad(self): super().zero_grad()
        
def acc_create_opt(self, lr:Floats, wd:Floats=0.):
        "Create optimizer with `lr` learning rate and `wd` weight decay."
        self.opt = AccumulateOptimWrapper.create(self.opt_func, lr, self.layer_groups,
                                         wd=wd, true_wd=self.true_wd, bn_wd=self.bn_wd)
Learner.create_opt = acc_create_opt   

@dataclass
class AccumulateStep(LearnerCallback):
    """
    Does accumlated step every nth step by accumulating gradients
    """
    def __init__(self, learn:Learner, n_step:int = 1):
        super().__init__(learn)
        self.n_step = n_step

    def on_epoch_begin(self, **kwargs):
        "init samples and batches, change optimizer"
        self.acc_batches = 0
        
    def on_batch_begin(self, last_input, last_target, **kwargs):
        "accumulate samples and batches"
        self.acc_batches += 1
        
    def on_backward_end(self, **kwargs):
        "step if number of desired batches accumulated, reset samples"
        if (self.acc_batches % self.n_step) == self.n_step - 1:
            for p in (self.learn.model.parameters()):
                if p.requires_grad: p.grad.div_(self.acc_batches)
    
            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0
    
    def on_epoch_end(self, **kwargs):
        "step the rest of the accumulated grads"
        if self.acc_batches > 0:
            for p in (self.learn.model.parameters()):
                if p.requires_grad: p.grad.div_(self.acc_batches)
            self.learn.opt.real_step()
            self.learn.opt.real_zero_grad()
            self.acc_batches = 0
            
