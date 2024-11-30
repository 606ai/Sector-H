import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from typing import Optional, Dict, Any, Callable
import os
import logging
from pathlib import Path
import json
from ..config import get_settings
from .training import ModelTrainer
from .experiment import ExperimentTracker

logger = logging.getLogger(__name__)

class DistributedTrainer:
    """Handles distributed training across multiple GPUs/machines."""
    
    def __init__(
        self,
        model_fn: Callable,
        optimizer_fn: Callable,
        criterion_fn: Callable,
        experiment_name: str,
        world_size: Optional[int] = None
    ):
        self.settings = get_settings()
        self.model_fn = model_fn
        self.optimizer_fn = optimizer_fn
        self.criterion_fn = criterion_fn
        self.experiment_name = experiment_name
        self.world_size = world_size or torch.cuda.device_count()
        
        # Create output directory
        self.output_dir = Path("distributed_training") / experiment_name
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def setup(self, rank: int, world_size: int):
        """Initialize distributed training environment."""
        os.environ['MASTER_ADDR'] = self.settings.master_addr
        os.environ['MASTER_PORT'] = self.settings.master_port
        
        # Initialize process group
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=world_size,
            rank=rank
        )
        
        # Set device for this process
        torch.cuda.set_device(rank)
    
    def cleanup(self):
        """Clean up distributed training environment."""
        dist.destroy_process_group()
    
    def train(
        self,
        rank: int,
        world_size: int,
        dataset,
        val_dataset,
        run_params: Optional[Dict[str, Any]] = None
    ):
        """Training process for each GPU."""
        try:
            self.setup(rank, world_size)
            
            # Create model and move to GPU
            model = self.model_fn().to(rank)
            model = DDP(model, device_ids=[rank])
            
            # Create optimizer and criterion
            optimizer = self.optimizer_fn(model.parameters())
            criterion = self.criterion_fn()
            
            # Create data samplers
            train_sampler = DistributedSampler(
                dataset,
                num_replicas=world_size,
                rank=rank
            )
            val_sampler = DistributedSampler(
                val_dataset,
                num_replicas=world_size,
                rank=rank
            )
            
            # Create data loaders
            train_loader = DataLoader(
                dataset,
                batch_size=self.settings.batch_size,
                sampler=train_sampler,
                num_workers=self.settings.num_workers,
                pin_memory=True
            )
            
            val_loader = DataLoader(
                val_dataset,
                batch_size=self.settings.batch_size,
                sampler=val_sampler,
                num_workers=self.settings.num_workers,
                pin_memory=True
            )
            
            # Create trainer and experiment tracker
            trainer = ModelTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=f"cuda:{rank}",
                experiment_name=f"{self.experiment_name}_rank{rank}"
            )
            
            # Only track experiments on rank 0
            if rank == 0:
                tracker = ExperimentTracker(self.experiment_name)
                tracker.start_run(run_params or {})
            else:
                tracker = None
            
            # Train model
            best_metrics, model_path = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                num_epochs=self.settings.num_epochs,
                run_params=run_params
            )
            
            # Save distributed training results
            if rank == 0:
                results = {
                    "world_size": world_size,
                    "best_metrics": best_metrics,
                    "model_path": str(model_path),
                    "run_params": run_params
                }
                
                results_path = self.output_dir / "results.json"
                with results_path.open("w") as f:
                    json.dump(results, f, indent=2)
            
            dist.barrier()
            self.cleanup()
            
        except Exception as e:
            logger.error(f"Error in distributed training process {rank}: {str(e)}")
            raise
    
    def run_distributed(
        self,
        dataset,
        val_dataset,
        run_params: Optional[Dict[str, Any]] = None
    ):
        """Launch distributed training processes."""
        try:
            mp.spawn(
                self.train,
                args=(self.world_size, dataset, val_dataset, run_params),
                nprocs=self.world_size,
                join=True
            )
        except Exception as e:
            logger.error(f"Failed to launch distributed training: {str(e)}")
            raise

class DistributedInference:
    """Handles distributed inference across multiple GPUs."""
    
    def __init__(
        self,
        model_fn: Callable,
        world_size: Optional[int] = None
    ):
        self.settings = get_settings()
        self.model_fn = model_fn
        self.world_size = world_size or torch.cuda.device_count()
    
    def inference(
        self,
        rank: int,
        world_size: int,
        data_loader: DataLoader
    ) -> torch.Tensor:
        """Run inference on each GPU."""
        try:
            # Initialize process group
            dist.init_process_group(
                backend='nccl',
                init_method='env://',
                world_size=world_size,
                rank=rank
            )
            
            # Create model and move to GPU
            model = self.model_fn().to(rank)
            model = DDP(model, device_ids=[rank])
            model.eval()
            
            predictions = []
            with torch.no_grad():
                for batch in data_loader:
                    output = model(batch.to(rank))
                    predictions.append(output)
            
            # Gather predictions from all processes
            all_predictions = [torch.zeros_like(torch.cat(predictions))
                             for _ in range(world_size)]
            dist.all_gather(all_predictions, torch.cat(predictions))
            
            dist.destroy_process_group()
            return torch.cat(all_predictions)
            
        except Exception as e:
            logger.error(f"Error in distributed inference process {rank}: {str(e)}")
            raise
    
    def run_distributed_inference(
        self,
        dataset,
        batch_size: Optional[int] = None
    ) -> torch.Tensor:
        """Launch distributed inference processes."""
        try:
            # Create data loader
            data_loader = DataLoader(
                dataset,
                batch_size=batch_size or self.settings.batch_size,
                num_workers=self.settings.num_workers,
                pin_memory=True
            )
            
            # Launch processes
            predictions = mp.spawn(
                self.inference,
                args=(self.world_size, data_loader),
                nprocs=self.world_size,
                join=True
            )
            
            return predictions
            
        except Exception as e:
            logger.error(f"Failed to launch distributed inference: {str(e)}")
            raise
