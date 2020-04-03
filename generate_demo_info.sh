# python generate_demo_by_info.py --algo ppo2 --env QbertNoFrameskip-v4 --deterministic
# python generate_demo_by_info.py --algo ppo2 --env PongNoFrameskip-v4 --deterministic
# python generate_demo_by_info.py --algo ppo2 --env BreakoutNoFrameskip-v4 --no-deterministic
# python generate_demo_by_info.py --algo ppo2 --env BeamRiderNoFrameskip-v4 --deterministic
python generate_demo_by_info.py --algo ppo2 --env EnduroNoFrameskip-v4  --n-episodes 5 --deterministic
# python generate_demo_by_info.py --algo ppo2 --env SeaquestNoFrameskip-v4 --deterministic
# python generate_demo_by_info.py --algo ppo2 --env SpaceInvadersNoFrameskip-v4 --deterministic

# source activate stable-baselines
# export PYTHONPATH=$PWD:$PYTHONPATH