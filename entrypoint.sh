#!/bin/bash
set -e

# setup ros2 environment
source "/opt/gym_robo_repos/install/setup.bash"

exec "$@"
