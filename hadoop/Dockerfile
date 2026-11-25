FROM apache/hadoop:3.5.0

ENV HADOOP_HOME=/opt/hadoop
ENV PATH="/opt/hadoop/bin:${PATH}"

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 \
    && rm -rf /var/lib/apt/lists/* \
    && install -d -o hadoop /var/lib/hadoop/name /var/lib/hadoop/data
COPY core-site.xml hdfs-site.xml mapred-site.xml yarn-site.xml /opt/hadoop/etc/hadoop/
USER hadoop
