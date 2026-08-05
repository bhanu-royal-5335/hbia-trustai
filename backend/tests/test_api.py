import pytest
from httpx import AsyncClient, ASGITransport
from app.main import app
from app.core.database import init_db


@pytest.fixture(autouse=True)
async def setup_db():
    await init_db()


@pytest.mark.asyncio
async def test_health_check():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        response = await ac.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


@pytest.mark.asyncio
async def test_auth_registration_and_login():
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        import uuid
        email = f"user_{uuid.uuid4().hex[:6]}@example.com"
        password = "TestPassword123!"

        # Register
        reg_resp = await ac.post(
            "/api/v1/auth/register",
            json={"email": email, "password": password, "full_name": "Test User"},
        )
        assert reg_resp.status_code == 200
        user_data = reg_resp.json()
        assert user_data["email"] == email

        # Login
        login_resp = await ac.post(
            "/api/v1/auth/login",
            data={"username": email, "password": password},
        )
        assert login_resp.status_code == 200
        token_data = login_resp.json()
        assert "access_token" in token_data

        # Profile (Me)
        me_resp = await ac.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {token_data['access_token']}"},
        )
        assert me_resp.status_code == 200
        assert me_resp.json()["email"] == email
