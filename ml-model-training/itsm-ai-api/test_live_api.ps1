# Test Script for ITSM AI API on HuggingFace Spaces
# API URL: https://srujanreddynadipi-itsm-ai-api.hf.space

$API_URL = "https://srujanreddynadipi-itsm-ai-api.hf.space"

Write-Host "ITSM AI API Test Suite" -ForegroundColor Cyan
Write-Host "======================" -ForegroundColor Cyan
Write-Host ""

# Test 1: Health Check
Write-Host "Test 1: Health Check" -ForegroundColor Yellow
Write-Host "--------------------"
try {
    $response = Invoke-WebRequest -Uri "$API_URL/health" -UseBasicParsing
    $health = $response.Content | ConvertFrom-Json
    Write-Host "Status: $($health.status)" -ForegroundColor Green
    Write-Host "Models Loaded: $($health.models_loaded)" -ForegroundColor Green
    Write-Host "Response:" -ForegroundColor Gray
    $health | ConvertTo-Json -Depth 5
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""

# Test 2: Predict Ticket - Hardware Issue
Write-Host "Test 2: Predict Ticket - Hardware Issue (Laptop Not Starting)" -ForegroundColor Yellow
Write-Host "---------------------------------------------------------------"
$ticket1 = @{
    user = "john.doe@company.com"
    title = "Laptop won't start"
    description = "My laptop is not turning on. I tried pressing the power button multiple times but nothing happens."
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "$API_URL/predict" -Method Post -Body $ticket1 -ContentType "application/json" -UseBasicParsing
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Category: $($result.category)" -ForegroundColor Green
    Write-Host "Priority: $($result.priority)" -ForegroundColor Green
    Write-Host "Resolver Group: $($result.resolver_group)" -ForegroundColor Green
    Write-Host "Confidence: $($result.confidence)%" -ForegroundColor Green
    Write-Host ""
    Write-Host "Full Response:" -ForegroundColor Gray
    $result | ConvertTo-Json -Depth 5
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""
Write-Host ""

# Test 3: Predict Ticket - Software Issue
Write-Host "Test 3: Predict Ticket - Software Issue (Email Access)" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------"
$ticket2 = @{
    user = "jane.smith@company.com"
    title = "Cannot access email"
    description = "I am unable to log into my email account. Getting authentication error message."
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "$API_URL/predict" -Method Post -Body $ticket2 -ContentType "application/json" -UseBasicParsing
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Category: $($result.category)" -ForegroundColor Green
    Write-Host "Priority: $($result.priority)" -ForegroundColor Green
    Write-Host "Resolver Group: $($result.resolver_group)" -ForegroundColor Green
    Write-Host ""
    Write-Host "Reasoning:" -ForegroundColor Cyan
    Write-Host $result.reasoning
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""
Write-Host ""

# Test 4: Predict Ticket - Network Issue
Write-Host "Test 4: Predict Ticket - Network Issue (Internet Connection)" -ForegroundColor Yellow
Write-Host "--------------------------------------------------------------"
$ticket3 = @{
    user = "mike.jones@company.com"
    title = "No internet connection"
    description = "My computer shows connected to WiFi but I cannot access any websites or applications."
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "$API_URL/predict" -Method Post -Body $ticket3 -ContentType "application/json" -UseBasicParsing
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Category: $($result.category)" -ForegroundColor Green
    Write-Host "Priority: $($result.priority)" -ForegroundColor Green
    Write-Host "Resolver Group: $($result.resolver_group)" -ForegroundColor Green
    Write-Host "Impact: $($result.impact) | Urgency: $($result.urgency)" -ForegroundColor Cyan
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""
Write-Host ""

# Test 5: Find Similar Tickets
Write-Host "Test 5: Find Similar Tickets" -ForegroundColor Yellow
Write-Host "-----------------------------"
$similarQuery = @{
    title = "Printer not working"
    description = "Office printer shows error"
    top_k = 3
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "$API_URL/find-similar" -Method Post -Body $similarQuery -ContentType "application/json" -UseBasicParsing
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Found $($result.similar_tickets.Count) similar tickets:" -ForegroundColor Green
    foreach ($ticket in $result.similar_tickets) {
        Write-Host "  - Ticket: $($ticket.title) (Similarity: $([math]::Round($ticket.similarity_score * 100, 2))%)" -ForegroundColor Cyan
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""
Write-Host ""

# Test 6: Search Knowledge Base
Write-Host "Test 6: Search Knowledge Base" -ForegroundColor Yellow
Write-Host "------------------------------"
$kbQuery = @{
    query = "how to reset password"
    top_k = 3
} | ConvertTo-Json

try {
    $response = Invoke-WebRequest -Uri "$API_URL/search-kb" -Method Post -Body $kbQuery -ContentType "application/json" -UseBasicParsing
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Found $($result.results.Count) knowledge base articles:" -ForegroundColor Green
    foreach ($article in $result.results) {
        Write-Host "  $($article.rank). $($article.title) (Score: $([math]::Round($article.score, 3)))" -ForegroundColor Cyan
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""
Write-Host ""

# Test 7: Auto-Draft Response
Write-Host "Test 7: Auto-Draft Response with KB Integration" -ForegroundColor Yellow
Write-Host "-------------------------------------------------"
$draftReq = @{
    ticket_id = "INC001234"
    category = "Account_Access"
    description = "User cannot log into their account"
    kb_context = @(
        @{
            title = "Password Reset Procedure"
            content = "To reset your password, go to the login page and click 'Forgot Password'"
        }
    )
} | ConvertTo-Json -Depth 5

try {
    $response = Invoke-WebRequest -Uri "$API_URL/auto-draft" -Method Post -Body $draftReq -ContentType "application/json" -UseBasicParsing
    $result = $response.Content | ConvertFrom-Json
    Write-Host "Draft Response Generated:" -ForegroundColor Green
    Write-Host $result.draft_response -ForegroundColor White
    Write-Host ""
    Write-Host "Recommendations:" -ForegroundColor Cyan
    foreach ($rec in $result.recommendations) {
        Write-Host "  - $rec" -ForegroundColor Gray
    }
} catch {
    Write-Host "Error: $_" -ForegroundColor Red
}
Write-Host ""
Write-Host ""

Write-Host "======================" -ForegroundColor Cyan
Write-Host "All Tests Completed!" -ForegroundColor Green
Write-Host ""
Write-Host "API Documentation: $API_URL/docs" -ForegroundColor Yellow
Write-Host "ReDoc Documentation: $API_URL/redoc" -ForegroundColor Yellow
