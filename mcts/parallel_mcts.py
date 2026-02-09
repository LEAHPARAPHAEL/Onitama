import torch
import torch.multiprocessing as mp
import time

class InferenceServer:
    def __init__(self, model, device, max_batch_size=256, timeout=0.01):
        self.model = model
        self.device = device
        self.max_batch_size = max_batch_size
        self.timeout = timeout
        self.keep_running = True
        
        # Queues for communication
        # input_queue: stores (worker_id, nn_input)
        self.input_queue = mp.Queue()
        # output_queues: dictionary of queues, one per worker
        self.output_queues = {}

    def register_worker(self, worker_id):
        self.output_queues[worker_id] = mp.Queue()
        return self.input_queue, self.output_queues[worker_id]

    def run(self):
        print("Inference Server Started")
        self.model.eval()
        
        while self.keep_running:
            batch_inputs = []
            worker_ids = []
            
            # 1. Collect inputs (wait up to timeout to fill batch)
            start_time = time.time()
            while len(batch_inputs) < self.max_batch_size:
                try:
                    # Non-blocking check for new data
                    if self.input_queue.empty():
                        if len(batch_inputs) > 0 and (time.time() - start_time > self.timeout):
                            break # Send what we have
                        time.sleep(0.0001) # Sleep briefly to save CPU
                        continue
                        
                    w_id, nn_input = self.input_queue.get_nowait()
                    batch_inputs.append(nn_input)
                    worker_ids.append(w_id)
                    
                except Exception:
                    break

            if not batch_inputs:
                continue

            # 2. Run Inference (The only time GPU is used)
            inputs_tensor = torch.stack(batch_inputs).to(self.device)
            
            with torch.no_grad():
                logits, values = self.model(inputs_tensor)
                
                probs = torch.softmax(logits, dim=1).cpu()
                if self.model.wdl:
                    wdl = torch.softmax(values, dim=1)
                    vals = (wdl[:, 2] - wdl[:, 0]).cpu()
                else:
                    vals = values.cpu()

            # 3. Dispatch results back to workers
            for i, w_id in enumerate(worker_ids):
                result = (probs[i], vals[i])
                self.output_queues[w_id].put(result)


class ParallelMCTS:
    def __init__(self, config, input_queue, output_queue, worker_id):
        # ... copy your config init logic here ...
        # Note: We do NOT hold self.model anymore!
        
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.worker_id = worker_id
        
        # Hyperparameters
        mcts_config = config['mcts']
        self.num_simulations = mcts_config.get('simulations', 100)
        self.c_puct = mcts_config.get('c_puct', 1.0)
        # ... etc ...

    def search_batch(self, boards):
        batch_size = len(boards)
        roots = [MCTSNode() for _ in range(batch_size)]
        
        for _ in range(self.num_simulations):
            leaf_nodes = []
            valid_indices = []
            nn_inputs = []

            # --- SELECTION (Pure CPU) ---
            for i in range(batch_size):
                # ... same selection logic as your code ...
                # ... when you hit a leaf ...
                leaf_nodes.append(node)
                valid_indices.append(i)
                nn_inputs.append(get_nn_input(board))

            if not valid_indices:
                continue

            # --- INFERENCE REQUEST (IPC) ---
            # Send all inputs to server one by one (or you could implement batch send)
            for nn_input in nn_inputs:
                self.input_queue.put((self.worker_id, nn_input))
            
            # --- WAIT FOR ANSWERS ---
            # Block until we get exactly len(nn_inputs) answers back
            results = []
            for _ in range(len(nn_inputs)):
                results.append(self.output_queue.get())
            
            # --- BACKPROPAGATION (Pure CPU) ---
            for j, (probs, val) in enumerate(results):
                # ... same expansion logic as your code ...
                # note: probs and val are already on CPU
                node = leaf_nodes[j]
                node.expand(probs, legal_moves) # You need to handle masking here
                node.backpropagate(val.item())

        # ... return policies logic ...
        return policies