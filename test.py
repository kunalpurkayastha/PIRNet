import torch
from torch.utils.data import DataLoader
from model.unet import UNet
from data.dataset import MRIDataset
from config import get_args
from utils.visualization import plot_comparison

def test(model, test_loader, device):
    model.eval()
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            
            # Visualize results
            for i in range(data.size(0)):
                plot_comparison(data[i, 0].cpu().numpy(), 
                                output[i, 0].cpu().numpy())

def main():
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = UNet().to(device)
    model.load_state_dict(torch.load(f'{args.output_dir}/best_model.pth'))
    
    test_dataset = MRIDataset(args.data_dir, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size)
    
    test(model, test_loader, device)

if __name__ == '__main__':
    main()