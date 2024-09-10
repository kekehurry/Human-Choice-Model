# Human-Behavior-Model

# Init datasets

### create a neo4j docker container with the following command:

```
docker run \
    -it \
    --restart always \
    --publish=7474:7474 --publish=7687:7687 \
    --env NEO4J_AUTH=neo4j/neo4jgraph \
    --volume=$HOME/Documents/neo4j/data:/data \
    --volume=$HOME/Documents/neo4j/logs:/logs \
    --volume=$HOME/Documents/neo4j/conf:/conf \
    --env NEO4J_dbms_memory_pagecache_size=4G \
    --env NEO4J_PLUGINS='["apoc", "graph-data-science"]' \
    --name neo4j \
    neo4j:5.20.0

```

```
sudo chown 755 $HOME/Documents/neo4j/data
sudo chown 755 $HOME/Documents/neo4j/logs
sudo chown 755 $HOME/Documents/neo4j/conf
```
