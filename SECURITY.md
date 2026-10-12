# Reporting security issues

Use GitHub's private vulnerability reporting option when it is available for this repository. If it is unavailable, open an issue that requests a private contact without including an exploit, credentials or participant data.

The Flask service is intended for local research use and listens on the local machine by default. Do not expose it as a public service without adding authentication, request limits and deployment controls. Set credentials through environment variables. Dataset upload is an explicit action; the publication copies of the older examples require `HF_UPLOAD=1` before uploading.
