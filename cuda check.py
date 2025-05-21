import torch
import torch.version # 如果pytorch安装成功即可导入
print(torch.__version__)  #注意是双下划线
print(torch.cuda.is_available()) # 查看CUDA是否可用
print(torch.cuda.device_count()) # 查看可用的CUDA数量
print(torch.version.cuda) # 查看CUDA的版本号
