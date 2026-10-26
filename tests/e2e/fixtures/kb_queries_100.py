"""100 Evaluation queries for KB comparison benchmark (HARS vs jbcontext).

Structured into 8 representative categories:
  1. AWS (15 queries)
  2. GCP (12 queries)
  3. Kubernetes & Platform Controllers (22 queries)
  4. Monitoring & Observability (16 queries)
  5. Internal Services & Platforms (15 queries)
  6. Tools & Workflows (10 queries)
  7. Exact Identifiers & Code References (5 queries)
  8. Out-of-Domain & Negatives (5 queries)

Total: 100 queries
"""

KB_QUERIES_100: list[tuple[str, list[str], str]] = [
    # =========================================================================
    # 1. AWS Domain (15 queries)
    # =========================================================================
    ("What are the core AWS accounts and organization structure in JetBrains?", ["root", "organization", "account"], "aws-organization"),
    ("How is AWS ECR repository lifecycle and replication configured?", ["ecr", "repository", "lifecycle"], "aws-ecr"),
    ("What is the setup for AWS MSK Managed Kafka clusters?", ["msk", "kafka", "cluster"], "aws-msk"),
    ("How does Kafka Connect integrate with AWS MSK?", ["kafka", "connect", "plugin"], "aws-kafka-connect"),
    ("What is the AWS CloudFront CDN distribution setup and caching policy?", ["cloudfront", "distribution", "cache"], "aws-cloudfront"),
    ("How are AWS WAF web ACL rules and rate limits applied to ALBs?", ["waf", "acl", "rule"], "aws-waf"),
    ("What is the RDS PostgreSQL configuration, backup and multi-AZ policy?", ["rds", "postgres", "backup"], "aws-rds"),
    ("How is AWS Direct Connect configured for interconnecting on-prem and cloud?", ["direct connect", "gateway", "dx"], "aws-direct-connect"),
    ("What is the AWS KMS key management and rotation strategy?", ["kms", "key", "rotation"], "aws-kms"),
    ("How is Amazon SES used for transactional mail sending and domain verification?", ["ses", "email", "domain"], "aws-ses"),
    ("What is the shared Elasticsearch / OpenSearch cluster setup in AWS?", ["opensearch", "elasticsearch", "cluster"], "aws-shared-es"),
    ("What is CUDOS and how is AWS cost intelligence visualized?", ["cudos", "cost", "quicksight"], "aws-cudos"),
    ("How do we request a new AWS account or budget increase?", ["account", "budget", "request"], "aws-accounts"),
    ("What IAM roles and Service Control Policies are applied across AWS OUs?", ["scp", "ou", "iam"], "aws-scp-iam"),
    ("How are AWS Transit Gateway and VPC peering configured?", ["transit gateway", "vpc", "peering"], "aws-tgw-peering"),

    # =========================================================================
    # 2. GCP Domain (12 queries)
    # =========================================================================
    ("What are the storage tiers and minimum volume size for Google Cloud Filestore?", ["filestore", "100", "gib"], "gcp-filestore-tiers"),
    ("How is GCP Artifact Registry organized for Docker container images?", ["artifact registry", "docker", "repository"], "gcp-artifact-registry"),
    ("How does Identity-Aware Proxy (IAP) work for securing GCP resources?", ["iap", "proxy", "oauth"], "gcp-iap-auth"),
    ("What is the standard VPC and Shared VPC network topology in GCP?", ["shared vpc", "network", "subnetwork"], "gcp-shared-vpc"),
    ("What are the main GKE clusters located in europe-west regions?", ["gke", "europe-west", "cluster"], "gcp-gke-europe"),
    ("How is Cloud NAT configured for outbound internet access in GKE?", ["cloud nat", "nat", "gateway"], "gcp-cloud-nat"),
    ("How are GCP project folders, billing accounts and quotas managed?", ["project", "billing", "quota"], "gcp-projects-billing"),
    ("What is the procedure for provisioning a new GCP project for a team?", ["project", "provision", "team"], "gcp-project-provision"),
    ("How do we configure Workload Identity between GKE and GCP service accounts?", ["workload identity", "serviceaccount", "gsa"], "gcp-workload-identity"),
    ("What are the firewall rules and ingress filtering best practices in GCP?", ["firewall", "ingress", "rule"], "gcp-firewall-rules"),
    ("How is Cloud Storage (GCS) lifecycle and bucket versioning configured?", ["gcs", "bucket", "lifecycle"], "gcp-gcs-lifecycle"),
    ("What GKE node pools and autoscaling parameters are used in production?", ["nodepool", "autoscaling", "gke"], "gcp-gke-nodepools"),

    # =========================================================================
    # 3. Kubernetes Core & Controllers (22 queries)
    # =========================================================================
    ("What does the JIP controller do for Kubernetes namespace provisioning?", ["jip", "namespace", "quota"], "k8s-jip-controller"),
    ("How are namespace quotas and CPU/RAM limits enforced in K8s clusters?", ["quota", "limitrange", "resourcequota"], "k8s-quotas-limits"),
    ("How is Cert-Manager configured with Let's Encrypt ClusterIssuers?", ["cert-manager", "clusterissuer", "encrypt"], "k8s-cert-manager"),
    ("What is the External Secrets Operator setup with HashiCorp Vault?", ["external-secrets", "vault", "secretstore"], "k8s-external-secrets"),
    ("How is ArgoCD structured for GitOps deployments and ApplicationSets?", ["argocd", "gitops", "application"], "k8s-argocd-gitops"),
    ("What is the ArgoCD security model and destination cluster registration?", ["argocd", "cluster", "destination"], "k8s-argocd-clusters"),
    ("How are Istio VirtualServices and Ingress Gateways configured?", ["istio", "virtualservice", "gateway"], "k8s-istio-ingress"),
    ("What is the Istio mTLS and sidecar injection policy across namespaces?", ["istio", "mtls", "sidecar"], "k8s-istio-mtls"),
    ("How does the Authorization Controller handle RBAC and group bindings?", ["authorization", "rbac", "binding"], "k8s-auth-controller"),
    ("How are NVIDIA GPUs provisioned and time-sliced in Kubernetes nodes?", ["gpu", "nvidia", "operator"], "k8s-gpu-nvidia"),
    ("What is the Kro / Composition Controller used for in K8s?", ["kro", "composition", "crd"], "k8s-kro-composition"),
    ("How does the Dump Collector capture and upload application core dumps?", ["dump", "collector", "s3"], "k8s-dump-collection"),
    ("What does the Database Controller do for automated DB provisioning?", ["database", "controller", "provision"], "k8s-database-controller"),
    ("How is the Glue Controller used for AWS IAM role integration in K8s?", ["glue", "iam", "role"], "k8s-glue-controller"),
    ("What is the Knative Serving architecture and scale-to-zero mechanism?", ["knative", "serving", "scale"], "k8s-knative-serving"),
    ("How does the Altinity ClickHouse Operator deploy ClickHouse clusters?", ["clickhouse", "operator", "altinity"], "k8s-clickhouse-operator"),
    ("How is Kyverno policy engine used for validating and mutating K8s objects?", ["kyverno", "policy", "validate"], "k8s-kyverno-policies"),
    ("What is the procedure for transferring Kubernetes namespace ownership?", ["namespace", "ownership", "transfer"], "k8s-namespace-transfer"),
    ("How to query Kubernetes API audit logs in Loki or Cloud Logging?", ["audit", "log", "apiserver"], "k8s-audit-logs"),
    ("How are on-prem Kubernetes clusters bootstrapped and maintained?", ["on-prem", "cluster", "node"], "k8s-onprem-clusters"),
    ("What is the credential rotation controller responsible for?", ["credential", "rotation", "secret"], "k8s-credential-rotation"),
    ("How do we deploy an application to Kubernetes using standard templates?", ["helm", "deployment", "template"], "k8s-app-deployment"),

    # =========================================================================
    # 4. Monitoring & Observability (16 queries)
    # =========================================================================
    ("What is the architecture of Thanos: Query, Store, Sidecar and Compact?", ["thanos", "sidecar", "store"], "mon-thanos-arch"),
    ("How is Thanos Compact configured for downsampling and block retention?", ["thanos", "compact", "downsample"], "mon-thanos-compact"),
    ("How does VictoriaMetrics vmagent collect and forward Prometheus metrics?", ["victoriametrics", "vmagent", "scrape"], "mon-victoriametrics"),
    ("What is the Grafana provisioning model for dashboards and data sources as code?", ["grafana", "dashboard", "provision"], "mon-grafana-provision"),
    ("How are AlertManager routing trees, receivers and inhibition rules defined?", ["alertmanager", "receiver", "route"], "mon-alertmanager-routing"),
    ("What is the retention period and query limits for Loki log storage?", ["loki", "retention", "log"], "mon-loki-retention"),
    ("How to write LogQL queries in Loki for Kubernetes pod logs and events?", ["logql", "loki", "stream"], "mon-loki-logql"),
    ("What is the Prometheus metric naming and alerting convention in Pier?", ["alert", "metric", "prometheus"], "mon-prometheus-conventions"),
    ("How are Thanos Ruler and PrometheusRule CRDs deployed for alerting?", ["thanos", "ruler", "prometheusrule"], "mon-thanos-ruler"),
    ("What dashboards exist for monitoring Kubernetes node and cluster capacity?", ["dashboard", "node", "capacity"], "mon-node-dashboards"),
    ("How does OpenSearch / Logstash pipeline process application logs?", ["opensearch", "logstash", "pipeline"], "mon-opensearch-logs"),
    ("What is the on-call shift rotation and alert escalation workflow?", ["on-call", "rotation", "escalation"], "mon-oncall-workflow"),
    ("How to create a silence for an ongoing maintenance in AlertManager?", ["silence", "alertmanager", "maintenance"], "mon-alertmanager-silence"),
    ("What SLA/SLO metrics and error budget alerts are tracked for core services?", ["slo", "sli", "budget"], "mon-slo-sli"),
    ("How is blackbox-exporter used for external HTTP probe monitoring?", ["blackbox", "probe", "http"], "mon-blackbox-probes"),
    ("What is the Grafana role-based access control (RBAC) and team sync?", ["grafana", "rbac", "team"], "mon-grafana-rbac"),

    # =========================================================================
    # 5. Internal Services & Platforms (15 queries)
    # =========================================================================
    ("What is the LiteLLM budget limit per user per month and how to increase it?", ["1000", "USD", "month"], "srv-litellm-budget"),
    ("How do users obtain API keys and access models on LiteLLM?", ["key", "virtual", "model"], "srv-litellm-keys"),
    ("What is the role of Langfuse in tracing LiteLLM requests?", ["langfuse", "trace", "litellm"], "srv-langfuse-tracing"),
    ("Where are self-hosted LLM models running on Hetzner H200 GPU nodes?", ["hetzner", "h200", "vllm"], "srv-hetzner-h200"),
    ("What is the Harbor evaluation platform and how does it use gVisor sandbox?", ["harbor", "gvisor", "eval"], "srv-harbor-gvisor"),
    ("How does ZenML orchestrate AI Assistant evaluation pipelines on GKE?", ["zenml", "pipeline", "gke"], "srv-zenml-evals"),
    ("What is the NetBox IPAM and DCIM inventory used for in JetBrains infra?", ["netbox", "ipam", "prefix"], "srv-netbox-ipam"),
    ("How do Platform Tenants (DWS) request namespace quotas and storage?", ["tenant", "namespace", "quota"], "srv-platform-tenants"),
    ("What is the HashiCorp Vault deployment model and AppRole authentication?", ["vault", "approle", "secret"], "srv-vault-approle"),
    ("How does JFrog Artifactory mirror PyPI, npm and Maven repositories?", ["artifactory", "mirror", "repository"], "srv-artifactory-mirrors"),
    ("What machine learning compute resources are available on Nebius and Bizon?", ["nebius", "bizon", "gpu"], "srv-ml-compute"),
    ("How is Cloud Console used for managing internal service deployments?", ["cloud console", "deploy", "service"], "srv-cloud-console"),
    ("What are the guidelines for registering new dynamic models in LiteLLM?", ["model", "post", "register"], "srv-litellm-models"),
    ("How is the Saviynt identity governance integration configured?", ["saviynt", "identity", "access"], "srv-saviynt-identity"),
    ("What is the JetBrains Central (JCP) coding agent infrastructure?", ["jcp", "central", "agent"], "srv-jcp-central"),

    # =========================================================================
    # 6. Tools & Workflows (10 queries)
    # =========================================================================
    ("How to set up and run jbcontext index for semantic KB searching?", ["jbcontext", "index", "search"], "tool-jbcontext-index"),
    ("How is Skaffold used for building and deploying microservices to K8s?", ["skaffold", "build", "deploy"], "tool-skaffold-workflow"),
    ("What are the SRE YouTrack knowledge base articles SRE-A mirrored in repo?", ["sre-a", "youtrack", "article"], "tool-youtrack-sre-a"),
    ("How does TeamCity run automated CI/CD pipelines for infra repositories?", ["teamcity", "pipeline", "build"], "tool-teamcity-cicd"),
    ("What is the Pier support bot architecture and skills system?", ["pier", "skill", "agent"], "tool-pier-skills"),
    ("How to configure local dev environment using setup.md instructions?", ["setup", "environment", "step"], "tool-setup-guide"),
    ("What are the abbreviations and acronyms used across SRE documentation?", ["abbreviation", "acronym", "sre"], "tool-abbreviations"),
    ("How to write and maintain new skills in the .agents/skills directory?", ["skill", "creator", "guidelines"], "tool-skill-creator"),
    ("What is the Git workflow and submodule management strategy in Pier?", ["submodule", "repo", "git"], "tool-git-workflow"),
    ("How do we draft stakeholder notifications for breaking infra changes?", ["stakeholder", "notification", "breaking"], "tool-stakeholder-notify"),

    # =========================================================================
    # 7. Exact Identifiers & Specific Lookups (5 queries)
    # =========================================================================
    ("LiteLLM per-user budget 1000 USD", ["1000", "USD", "budget"], "exact-litellm-1000"),
    ("runtimeClassName: gvisor", ["gvisor", "runtime"], "exact-gvisor-runtime"),
    ("POST /model/new endpoint in LiteLLM", ["model", "new", "endpoint"], "exact-post-model-new"),
    ("SRE-A-529 SRE Stakeholder Analysis", ["stakeholder", "analysis", "529"], "exact-sre-a-529"),
    ("https://litellm.labs.jb.gg URL endpoints", ["litellm", "labs", "jb"], "exact-litellm-url"),

    # =========================================================================
    # 8. Out-of-Domain / Negative Queries (5 queries)
    # =========================================================================
    ("What is the capital of France and what are its main tourist attractions?", [], "ood-geography-paris"),
    ("How to make classic Italian pizza dough from scratch?", [], "ood-cooking-pizza"),
    ("Explain the quantum electrodynamics Lagrangian and gauge invariance.", [], "ood-physics-qed"),
    ("Which movie had the highest global box office revenue in 2024?", [], "ood-cinema-boxoffice"),
    ("What is the current market stock price of Apple Inc AAPL?", [], "ood-finance-stocks"),
]
