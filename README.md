### Active Vision
Docker configuration. Make sure you have install docker and docker nvidia toolkit before you run the following command.
```
printf "DOCKER_UID=$(id -u $USER)\nDOCKER_GID=$(id -g $USER)\nDOCKER_USER=$USER\n" > .env
docker compose up --build -d && docker exec -it metasim_pw bash
```
then inside the container
```
pip install -e humanoid_visualrl/rsl-rl-lib
```
