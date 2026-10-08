# Deploying `lambda_function.py` to AWS Lambda + API Gateway

Step-by-step guide to deploy the live prediction endpoint. Assumes you've already
run `data_pipeline.py` and `train_model.py` for at least one ticker, so a trained
model exists at `s3://your-bucket-name/stock-predictor/{TICKER}_xgb_model.joblib`.

Commands below use the AWS CLI v2. Replace `<...>` placeholders (account ID,
region, bucket name, etc.) with your own values.

---

## 0. Prerequisites

- AWS CLI v2 installed and configured (`aws configure`)
- Python 3.12 installed locally (matches the Lambda runtime — dependencies built
  on a different Python/OS combo can fail to import at runtime, especially
  binary packages like `numpy`/`xgboost`)
- Your account ID and preferred region handy:

```bash
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export AWS_REGION=us-west-2   # or your region
export BUCKET=your-bucket-name
```

---

## 1. Package the code + dependencies into a zip

`lambda_function.py` only needs a subset of `requirements.txt` at inference
time — training-only packages (`matplotlib`, `seaborn`, `scikit-learn` for
oversampling, `imbalanced-learn`) aren't imported by the handler and would
needlessly bloat the deployment package. Use a slimmer, inference-only list:

```bash
cat > requirements-lambda.txt << 'EOF'
joblib
numpy
pandas
requests
yfinance
nltk
xgboost
scikit-learn
EOF
```

> `scikit-learn` is listed even though `lambda_function.py` doesn't import it
> directly — `XGBClassifier` (the class `joblib.load()` reconstructs) is built
> on scikit-learn's estimator interface, so it must be importable for the
> pickled model to load.

Build the package:

```bash
mkdir -p build/package
pip install -r requirements-lambda.txt -t build/package --platform manylinux2014_x86_64 \
    --python-version 3.12 --only-binary=:all:
cp lambda_function.py build/package/

cd build/package
zip -r ../lambda_deploy.zip .
cd ../..
```

> **`--platform`/`--only-binary` matter** if you're building on macOS or an ARM
> Mac: Lambda's default runtime is `x86_64` Linux, so pip must fetch Linux
> wheels rather than compiling/linking against your local OS. Add
> `--architectures arm64` to the `create-function` call below (and drop
> `--platform`/use `manylinux2014_aarch64`) if you'd rather run on Graviton.

> **Size check:** `boto3`/`botocore` are preinstalled in the Lambda Python
> runtime, so they're deliberately excluded above. Even so, `pandas` +
> `numpy` + `xgboost` + `scikit-learn` together can approach Lambda's
> **250 MB unzipped** deployment package limit. If you hit it: strip
> `**/tests`, `**/*.dist-info`, and `**/__pycache__` from `build/package`
> before zipping, or switch to a **container image** deployment (`docker
> build` off `public.ecr.aws/lambda/python:3.12`, push to ECR, and point
> `create-function` at the image instead of a zip) — container images get a
> 10 GB budget instead of 250 MB.

---

## 2. Create the IAM execution role

```bash
cat > trust-policy.json << 'EOF'
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Principal": { "Service": "lambda.amazonaws.com" },
    "Action": "sts:AssumeRole"
  }]
}
EOF

aws iam create-role \
  --role-name stock-predictor-lambda-role \
  --assume-role-policy-document file://trust-policy.json

# CloudWatch Logs permissions
aws iam attach-role-policy \
  --role-name stock-predictor-lambda-role \
  --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole

# Read-only access to the model/metrics objects in your bucket
cat > s3-read-policy.json << EOF
{
  "Version": "2012-10-17",
  "Statement": [{
    "Effect": "Allow",
    "Action": "s3:GetObject",
    "Resource": "arn:aws:s3:::${BUCKET}/stock-predictor/*"
  }]
}
EOF

aws iam put-role-policy \
  --role-name stock-predictor-lambda-role \
  --policy-name stock-predictor-s3-read \
  --policy-document file://s3-read-policy.json

export ROLE_ARN=arn:aws:iam::${AWS_ACCOUNT_ID}:role/stock-predictor-lambda-role
```

IAM roles can take a few seconds to propagate — if `create-function` in the
next step fails with an assume-role error, wait ~10s and retry.

---

## 3. Create the Lambda function

```bash
aws lambda create-function \
  --function-name stock-predictor \
  --runtime python3.12 \
  --handler lambda_function.lambda_handler \
  --role "$ROLE_ARN" \
  --zip-file fileb://build/lambda_deploy.zip \
  --timeout 30 \
  --memory-size 1024 \
  --region "$AWS_REGION"
```

- **Timeout 30s** — a cold-start prediction chains a yfinance download, an
  optional Finnhub call, VADER lexicon loading, and model inference; the
  default 3s timeout will not be enough.
- **Memory 1024 MB** — `pandas`/`numpy`/`xgboost` are memory-hungry to import;
  under-provisioning shows up as slow cold starts more than outright failures.

## 4. Set environment variables

```bash
aws lambda update-function-configuration \
  --function-name stock-predictor \
  --environment "Variables={S3_BUCKET=$BUCKET,FINNHUB_API_KEY=YOUR_FINNHUB_KEY}" \
  --region "$AWS_REGION"
```

- `S3_BUCKET` — required; matches `--bucket` used with `data_pipeline.py`/`train_model.py`.
- `FINNHUB_API_KEY` — optional. If omitted, `sentiment_score` falls back to
  `0.0` for every prediction (see `add_sentiment_features()` in
  `lambda_function.py`) rather than the function failing.
- `MODEL_KEY` — optional; only set this if you want to pin a specific model
  object instead of the per-ticker default
  (`stock-predictor/{TICKER}_xgb_model.joblib`).

> Both `yfinance` and the Finnhub request need outbound internet access. A
> Lambda function **not** attached to a VPC has this by default; if you later
> attach it to a VPC (e.g. to reach an RDS instance), you'll also need a NAT
> gateway or these calls will start timing out.

---

## 5. Create an HTTP API Gateway route

```bash
export LAMBDA_ARN=$(aws lambda get-function --function-name stock-predictor \
    --query 'Configuration.FunctionArn' --output text --region "$AWS_REGION")

# 5a. Create the HTTP API
export API_ID=$(aws apigatewayv2 create-api \
  --name stock-predictor-api \
  --protocol-type HTTP \
  --query 'ApiId' --output text --region "$AWS_REGION")

# 5b. Create the Lambda proxy integration
export INTEGRATION_ID=$(aws apigatewayv2 create-integration \
  --api-id "$API_ID" \
  --integration-type AWS_PROXY \
  --integration-uri "$LAMBDA_ARN" \
  --payload-format-version 2.0 \
  --query 'IntegrationId' --output text --region "$AWS_REGION")

# 5c. Route POST /predict to that integration
aws apigatewayv2 create-route \
  --api-id "$API_ID" \
  --route-key "POST /predict" \
  --target "integrations/$INTEGRATION_ID" \
  --region "$AWS_REGION"

# 5d. Auto-deploying default stage
aws apigatewayv2 create-stage \
  --api-id "$API_ID" \
  --stage-name '$default' \
  --auto-deploy \
  --region "$AWS_REGION"

# 5e. Let API Gateway invoke the function
aws lambda add-permission \
  --function-name stock-predictor \
  --statement-id apigateway-invoke \
  --action lambda:InvokeFunction \
  --principal apigateway.amazonaws.com \
  --source-arn "arn:aws:execute-api:${AWS_REGION}:${AWS_ACCOUNT_ID}:${API_ID}/*/*/predict" \
  --region "$AWS_REGION"

echo "Endpoint: https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/predict"
```

---

## 6. Test it

```bash
curl -X POST "https://${API_ID}.execute-api.${AWS_REGION}.amazonaws.com/predict" \
  -H "Content-Type: application/json" \
  -d '{"ticker": "AAPL"}'
```

Expected response:

```json
{
  "ticker": "AAPL",
  "prediction": "UP",
  "confidence": 0.6134,
  "latest_close": 189.42,
  "latest_date": "2025-01-10",
  "model_version": "xgb_v1"
}
```

If it fails, check CloudWatch Logs first:

```bash
aws logs tail /aws/lambda/stock-predictor --follow --region "$AWS_REGION"
```

Common first-deploy issues:
- **`Unable to import module 'lambda_function'`** — a dependency was built for
  the wrong OS/architecture (see the `--platform` note in step 1).
- **`errorMessage: "No data returned for ticker '...'"`** — check the ticker
  symbol, and that the Lambda has outbound internet access (see the VPC note
  in step 4).
- **Timeout errors** — bump `--timeout` further; cold starts pulling in
  `pandas`/`xgboost` plus two outbound API calls can be slow on first
  invocation.

---

## 7. Redeploying after code changes

```bash
cp lambda_function.py build/package/
rm -f build/lambda_deploy.zip
cd build/package && zip -r ../lambda_deploy.zip . && cd ../..

aws lambda update-function-code \
  --function-name stock-predictor \
  --zip-file fileb://build/lambda_deploy.zip \
  --region "$AWS_REGION"
```

The dependency list (`requirements-lambda.txt`) rarely changes; you'll usually
only need to re-run the `cp` + `zip` + `update-function-code` sequence above
after editing `lambda_function.py`.
