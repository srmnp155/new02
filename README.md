# Sample Django Chatbot with Azure OpenAI

This Django app now includes:

- User sign up and login
- Profile management
- Password reset flow
- Login-protected chatbot
- Saved chat sessions per user
- Azure OpenAI access using endpoint + API key

## 1) Create virtual environment and install packages

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Configure environment variables

```powershell
Copy-Item .env.example .env
```

Edit `.env` and set:

- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`

## 3) Run Django

```powershell
python manage.py migrate
python manage.py runserver
```

Open `http://127.0.0.1:8000/` and test the chatbot.

## Verification flow

1. Open `http://127.0.0.1:8000/`
2. Sign up a new user
3. Login and open `/chat/`
4. Send messages, then create a new session with `New`
5. Confirm previous sessions appear in the left panel
6. Open `/profile/` and update account details
7. Trigger password reset from login/profile page

In development, password reset email content is printed to terminal (console backend).

## Azure App Service (GUI) production setup

Set these App Settings in Azure Portal:

- `DJANGO_SECRET_KEY`
- `DJANGO_DEBUG=false`
- `DJANGO_ALLOWED_HOSTS=<yourapp>.azurewebsites.net`
- `DJANGO_CSRF_TRUSTED_ORIGINS=https://<yourapp>.azurewebsites.net`
- `AZURE_OPENAI_ENDPOINT`
- `AZURE_OPENAI_API_KEY`
- `AZURE_OPENAI_API_VERSION`
- `AZURE_OPENAI_DEPLOYMENT`

Startup command:

```bash
python manage.py migrate && python manage.py collectstatic --noinput && gunicorn chatbot_project.wsgi --bind=0.0.0.0 --timeout 600
```
