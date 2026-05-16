FROM grafana/grafana:13.0.1

COPY ./provisioning /etc/grafana/provisioning
COPY ./dashboards /var/lib/grafana/dashboards
COPY ./config.ini /etc/grafana/grafana.ini

USER root
COPY ./entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
USER grafana

ENTRYPOINT ["/entrypoint.sh"]