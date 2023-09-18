

def save_checkpoint(model, optimizer, scheduler, epoch, filename='checkpoint.pth'):
    """
    모델의 체크포인트를 저장하는 함수.
    
    Parameters:
    - model (torch.nn.Module): 저장할 모델.
    - optimizer (torch.optim.Optimizer): 사용 중인 optimizer.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): 사용 중인 learning rate scheduler.
    - epoch (int): 현재 epoch.
    - filename (str): 저장할 체크포인트 파일의 이름.
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}.")

def load_checkpoint(model, optimizer, scheduler, filename='checkpoint.pth'):
    """
    체크포인트를 로드하여 모델, optimizer, scheduler에 적용하는 함수.
    
    Parameters:
    - model (torch.nn.Module): 체크포인트를 로드할 모델.
    - optimizer (torch.optim.Optimizer): 체크포인트를 로드할 optimizer.
    - scheduler (torch.optim.lr_scheduler._LRScheduler): 체크포인트를 로드할 scheduler.
    - filename (str): 로드할 체크포인트 파일의 이름.
    
    Returns:
    - epoch (int): 체크포인트의 epoch.
    - best_mIoU (float): 체크포인트의 최고 mIoU.
    """
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    print(f"Checkpoint loaded from {filename}.")
    return checkpoint['epoch'], checkpoint['best_mIoU']