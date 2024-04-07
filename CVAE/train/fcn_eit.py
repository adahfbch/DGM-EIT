"""
Stage2: Fully connected networks + encoder, encoder for the z and FCN for the reco_z
"""
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

from dataset_paper import dd_train,dd_val
from torch import optim
import tqdm
from fcn import *
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset

# device = torch.device("cuda")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def Re_sigma(x, reco_eit):
    Re_up = torch.nn.L1Loss()(x, reco_eit)
    t = torch.zeros(x.shape).to(device)
    Re_down = torch.nn.L1Loss()(x, t)
    Re_sigma = Re_up / Re_down
    return Re_sigma


epochs = 800
Learning_rate = [0.00008,0.0003]
Batch_size = [8]     #[8,16,32,48]

class MyDataset(Dataset):
    def __init__(self,dd):

        self.x_data = dd[0]
        self.y_data = dd[1]
        self.length = len(self.y_data)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


train_dataset = MyDataset(dd_train)
val_dataset=MyDataset(dd_val)


for batch_size in Batch_size:
    for learning_rate in Learning_rate:

        dd_train_loader = torch.utils.data.DataLoader(train_dataset, batch_size, shuffle=True, drop_last=True)
        dd_val_loader = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=False, drop_last=True)

        fcn_writer = SummaryWriter("path_{}_{}_{}".format(epochs, learning_rate, batch_size)) #change to your own path
        vae = torch.load("model.pt")  #change to your own model path
        vae.eval()
        fcn = MLP().to(device)
        optimizer = optim.Adam(fcn.parameters(), lr=learning_rate)


        for epoch in tqdm.trange(epochs, desc='Epoch Loop'):
            with tqdm.tqdm(dd_train_loader, total=dd_train_loader.batch_sampler.__len__()) as t:
                for idx, data in enumerate(dd_train_loader):
                    xs,ys = data
                    ys = ys.to(device)
                    xs = xs.to(device)

                    reco_z_train = fcn(ys)
                    z_train, mu_train, logvar_train = vae.encode(xs)
                    z_train = z_train.to(device)

                    train_loss = loss_fz(reco_z_train.squeeze(), z_train, batch_size)
                    optimizer.zero_grad()
                    train_loss.backward(retain_graph=True)
                    optimizer.step()
                    # fcn_writer.add_scalar("train_lr", optimizer.param_groups[0]["lr"], epoch)

                    t.set_description(f'Epoch {epoch}')
                    t.set_postfix(ordered_dict={'Loss': train_loss.item()})
                    t.update(1)

                fcn_writer.add_scalar("train_mse", train_loss, epoch)
                # loss_list.append(loss.item())
                # for name, layer in fcn.named_parameters():
                #     fcn_writer.add_histogram(name + '_grad', layer.grad.cpu().data.numpy(), epoch)
                #     fcn_writer.add_histogram(name + '_data', layer.cpu().data.numpy(), epoch)

                # test
                fcn.eval()
                with torch.no_grad():
                    for idx, data in enumerate(dd_val_loader):
                        xs, ys = data
                        ys = ys.to(device)
                        xs = xs.to(device)

                        reco_z_val = fcn(ys)
                        z_val, mu_val, logvar_val = vae.encode(xs)
                        val_loss = loss_fz(reco_z_val.squeeze(), z_val, batch_size)
                        Re_err = Re_sigma(z_val, reco_z_val.squeeze())
                        Abs_err = torch.nn.L1Loss()(z_val, reco_z_val.squeeze())

                    fcn_writer.add_scalar("val_mse", val_loss, epoch)
                    fcn_writer.add_scalar("Relative_error", Re_err, epoch)
                    fcn_writer.add_scalar("abs_error",Abs_err,epoch)
                    # test_loss_list.append(test_loss.item())

                    # if epoch % 10 == 0:
                    #     print('test Epoch : {} \tLoss: {:.5f}'.format(epoch, test_loss))

        torch.save(fcn, "path_{}_{}_{}.pt".format(epochs, learning_rate, batch_size)) #change to your own path
        fcn_writer.close()



