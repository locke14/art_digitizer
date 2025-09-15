Param(
  [Parameter(Mandatory=$true)][string]$ProjectId,
  [Parameter(Mandatory=$true)][string]$Region,
  [string]$ServiceName = "art-digitizer"
)

$Image = "gcr.io/$ProjectId/$ServiceName"

Write-Host "==> Setting gcloud project to $ProjectId"
gcloud config set project $ProjectId | Out-Null

Write-Host "==> Enabling required APIs (Run, Build)"
gcloud services enable run.googleapis.com cloudbuild.googleapis.com | Out-Null

Write-Host "==> Building image: $Image"
gcloud builds submit --tag $Image

Write-Host "==> Deploying to Cloud Run: $ServiceName in $Region"
gcloud run deploy $ServiceName `
  --image $Image `
  --platform managed `
  --allow-unauthenticated `
  --region $Region `
  --memory 1Gi

$Url = gcloud run services describe $ServiceName --region $Region --format='value(status.url)'
Write-Host "==> Deployed: $Url"

