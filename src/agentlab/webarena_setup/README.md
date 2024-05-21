# WebArena Setup


you can use our own ASW-hosted webarena servers (make sure that the servers are running w/ Massimo):
just set `SERVER_HOSTNAME` to e.g. `ec2-3-21-46-179.us-east-2.compute.amazonaws.com` in `check_webarena_servers.py`


if you haven't already, don't forget to run:
``` 
python -c "import nltk; nltk.download('punkt')"
```

## Do it yourself

(following https://github.com/web-arena-x/webarena/tree/main/environment_docker#pre-installed-amazon-machine-image)

We provide AMI which have all the websites pre-installed. You can use the AMI to start a new EC2 instance.

```
AMI Information: find in console, EC2 - AMI Catalog
Region: us-east-2
Name: webarena
ID: ami-06290d70feea35450
```

0. Create a security group that allows all inbound traffic.

1. Create an instance (recommended type: t3a.xlarge, 1000GB EBS root volume) from the webarena AMI. Use the security group just created and remember to select SSH key-pair.

2. Create an Elastic IP and bind to the instance to associate the instance with a static IP and hostname.

Take note of the hostname, usually in the form of "ec2-xx-xx-xx-xx.us-east-2.compute.amazonaws.com". We will refer to this as `SERVER_HOSTNAME`

4. connect to the instance a naviguate to `home/ubuntu/`

5. copy paste the `serve_webarena_aws.sh` and `serve_launch_webarena_aws.sh`

Set the `SERVER_HOSTNAME` variable in the `serve_webarena_aws.sh`

6. in a `tmux` of `screen`, launch `serve_webarena_aws.sh` or `repeat_webarena_aws.sh` if you want the web server to reset every hour.

7. come back the `ui-copilot` codebase. `SERVER_HOSTNAME` in `ui_copilot/src/llm/toolkit_configs` and `check_webarena_servers.py`.

## Important TODOs before the next WebArena experiments

1. We need to understand how evaluation corrupts the webarena servers.
Is it good enough to restart the docker servers every 1h, as in `repeat_webarena_aws.sh`?
Or are the databases outside the dockers? Then we would need to restart those as well.
Nuclear method is to fully relaunch the aws instance...


2. Also why, is `$SERVER_HOSTNAME:7565/` working on Toolkit but not `$SERVER_HOSTNAME:7565/admin`?
FYI, `7765` is a port w/ remapped to for Toolkit accessibility.

3. Gitlab interminently crashes. why?

4. Figure out if we are bottlenecked by Working Memory (hinted by Gasse)

NOTE: somehow, it seems like not binding the Elastic IP to the EC2 instance still works (I had forgotten to do it w/ the last instance I created) 
