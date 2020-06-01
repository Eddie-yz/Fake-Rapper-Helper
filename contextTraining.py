from RecognitionEncoder import DeepSpeech
import torch
from torch.nn import MSELoss
from torch.optim import Adam

def contextTraining(out_target, input_variable, input_sizes, rEncoder, epochs, criterion,optimizer):
    rEncoder.eval()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out_variable = rEncoder(input_variable, input_sizes)
        #print(out_variable[0][0][0][:5])
        loss = criterion(out_variable, out_target)
        loss = loss / input_target.size(0)  # average the loss by minibatch
        loss_value = loss.item()
        print("epoch:{}".format(epoch),loss_value)
        loss.backward()
        print(out_variable.grad, out_target.grad)
        optimizer.step()
            
def load_model(path):
    print("Loading state from model %s" % path)
    package = torch.load(path, map_location=lambda storage, loc: storage)
    model = DeepSpeech(audio_conf=package['audio_conf'])
    model.load_state_dict(package['state_dict'], strict=False)
    return model

if __name__ == "__main__":
    # load model parameters
    rEncoder = load_model('../deepspeech/librispeech_pretrained_v2.pth')
    device = torch.device("cuda")
    rEncoder = rEncoder.to(device)
    print(rEncoder.audio_conf)

    # TODO hyperparameters
    minibatch_size = 1
    freq_size = 16000
    max_seqlength = 100 
    lengths = 90        
    input_sizes = torch.LongTensor([lengths]*minibatch_size)
    
    # random variable
    from torch.autograd import Variable
    input_target = torch.randn(minibatch_size, 1, freq_size, max_seqlength, requires_grad=False).to(device) # TODO:target
    input_variable = Variable(torch.randn(input_target.shape).cuda(),requires_grad=True)
    
    criterion = MSELoss()
    optimizer = Adam([input_variable],lr=0.0001)
    
    rEncoder_target = load_model('../deepspeech/librispeech_pretrained_v2.pth')
    rEncoder_target = rEncoder_target.to(device)
    out_target = Variable(rEncoder_target(input_target, input_sizes))
    contextTraining(out_target, input_variable, input_sizes, rEncoder, 5, criterion, optimizer)
    