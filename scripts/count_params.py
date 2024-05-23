from prettytable import PrettyTable
from model.models import SiteLevelModel as model1
from model.models2 import SiteLevelModel as model2
from model.models3 import SiteLevelModel as model3
from model.models4 import SiteLevelModel as model4
from model.models5 import SiteLevelModel as model5
from model.models6 import SiteLevelModel as model6


def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

print(">>>>>>model RAW_SIGNAL --> BiLSTM 2layers, hidden_size=128, seq_len=5, signal_len=65 ====")
model_lstm2l = model3(model_type="raw_signal", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                      embedding_size=4, num_layers=2)
count_parameters(model_lstm2l)
print()
print(">>>>>>model RAW_SIGNAL --> BiLSTM 3layers, hidden_size=128, seq_len=5, signal_len=65 ====")
model_lstm3l = model3(model_type="raw_signal", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                      embedding_size=4, num_layers=3)
count_parameters(model_lstm3l)
print()
print(">>>>>>model RAW_SIGNAL --> BiLSTM 4layers, hidden_size=128, seq_len=5, signal_len=65 ====")
model_lstm4l = model3(model_type="raw_signal", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                      embedding_size=4, num_layers=4)
count_parameters(model_lstm4l)
print()
print(">>>>>>model RAW_SIGNAL --> ResNet (kernal_size=7, out_channels=(64,128,256,512)), signal_len=65 ====")
model_resnet_512k7 = model4(model_type="raw_signal", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                            embedding_size=4)
count_parameters(model_resnet_512k7)
print()
print(">>>>>>model RAW_SIGNAL --> ResNet (kernal_size=3, out_channels=(32,64,128,256)), signal_len=65 ====")
model_resnet_256k3 = model6(model_type="raw_signal", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                            embedding_size=4)
count_parameters(model_resnet_256k3)
print()
print(">>>>>>model RAW_SIGNAL --> ResNet (kernal_size=7, out_channels=(32,64,128,256)), signal_len=65 ====")
model_resnet_256k7 = model5(model_type="raw_signal", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                            embedding_size=4)
count_parameters(model_resnet_256k7)
print()
print(">>>>>>model RAW_SIGNAL --> ResNet (kernal_size=7, out_channels=(32,64,128,256)), signal_len=55 ====")
model_resnet_256k7_55 = model5(model_type="raw_signal", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=55,
                               embedding_size=4)
count_parameters(model_resnet_256k7_55)
print()
print(">>>>>>model RAW_SIGNAL --> ResNet (kernal_size=7, out_channels=(32,64,128,256)), signal_len=45 ====")
model_resnet_256k7_45 = model5(model_type="raw_signal", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=45,
                               embedding_size=4)
count_parameters(model_resnet_256k7_45)
print()


print(">>>>>>model BASECALL --> ResNet (kernal_size=7, out_channels=(64,128,256,512)) ====")
model_resnet_512k7 = model4(model_type="basecall", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                            embedding_size=4)
count_parameters(model_resnet_512k7)
print()
print(">>>>>>model BASECALL --> ResNet (kernal_size=7, out_channels=(32,64,128,256)) ====")
model_resnet_256k7 = model6(model_type="basecall", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                            embedding_size=4)
count_parameters(model_resnet_256k7)
print()
print(">>>>>>model BASECALL --> ResNet (kernal_size=3, out_channels=(32,64,128,256)) ====")
model_resnet_256k3 = model5(model_type="basecall", dropout_rate=0, hidden_size=128, seq_len=5, signal_lens=65,
                            embedding_size=4)
count_parameters(model_resnet_256k3)
print()
