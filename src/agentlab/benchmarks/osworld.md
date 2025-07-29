# Setup OSWorld in AgentLab

This guide walks you through setting up the OSWorld benchmark in AgentLab for GUI automation testing.

## Installation

1. **Clone and install OSWorld repository:**
   ```bash
   make osworld
   ```

2. **Complete OSWorld setup:**
   - Navigate to the `OSWorld/` directory
   - Follow the detailed setup instructions in the OSWorld README
   - Download required VM images and configure virtual machines


## Usage

### Entry Point Configuration

The main entry point `experiments/run_osworld.py` is currently configured with hardcoded parameters. To modify the execution:

1. **Edit the script directly** to change:
   - `n_jobs`: Number of parallel jobs (default: 4, set to 1 for debugging)
   - `use_vmware`: Set to `True` for VMware, `False` for other platforms
   - `relaunch`: Whether to continue incomplete studies
   - `agent_args`: List of agents to test (OSWORLD_CLAUDE, OSWORLD_OAI)
   - `test_set_name`: Choose between "test_small.json" or "test_all.json"

2. **Environment Variables:**
   - `AGENTLAB_DEBUG=1`: Automatically runs the debug subset (7 tasks from `osworld_debug_task_ids.json`)

### Task subsets

We provide different subsets of tasks:

- **Debug subset:** 7 tasks defined in `experiments/osworld_debug_task_ids.json` 
- **Small subset:** Tasks from `test_small.json`
- **Full subset:** All tasks from `test_all.json`

### Example Commands

```bash
# Run with default debug subset using sequential execution in VMware VM
python experiments/run_osworld.py
```

### Parallel Execution with Docker
To run OSWorld in parallel using Docker, ensure you have Docker installed and configured.
To install it, follow the section from the OSWorld README on [Docker setup](https://github.com/xlang-ai/OSWorld?tab=readme-ov-file#docker-server-with-kvm-support-for-better-performance).
Ensure that your docker installation support KVM, as OSWorld requires it for running VMs.
We also recommend pulling the latest Docker image for OSWorld before running the benchmark:

```bash
docker pull happysixd/osworld-docker
```

After setting up Docker, you can change the `use_vmware` parameter in the script to `False` and run:

```bash
python experiments/run_osworld.py
```
You can control number of parallel jobs by setting the `n_jobs` parameter in the script, the default is 4.
We recommend setting `n_jobs` to `your_number_of_cpu_cores - 2` to leave some resources for the host system and the benchmark itself.


### Configuration Notes

- **VMware path:** Currently hardcoded to `"OSWorld/vmware_vm_data/Ubuntu0/Ubuntu0.vmx"`
- **Parallel execution:** Automatically switches to sequential when using VMware
- **Relaunch capability:** Can continue incomplete studies by loading the most recent study

