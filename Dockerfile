FROM neo4j:5.23.0

# Set default local credentials for the Neo4j instance.
ENV NEO4J_AUTH=neo4j/password
ENV NEO4J_dbms_default__listen__address=0.0.0.0

# Expose both the HTTP and Bolt ports.
EXPOSE 7474 7687

# Persist data and logs outside the container.
VOLUME /data /logs /plugins

# Start Neo4j.
CMD ["neo4j"]
