"""Demo access: exchange the shared password for a bearer token."""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, Field

from app.core.config import Settings, get_settings
from app.core.security import make_token, password_matches, require_access

router = APIRouter(prefix="/auth", tags=["auth"])

SettingsDep = Annotated[Settings, Depends(get_settings)]


class LoginRequest(BaseModel):
    """Body of POST /api/auth/login."""

    password: str = Field(min_length=1, max_length=200)


class LoginResponse(BaseModel):
    """The bearer token to send as `Authorization: Bearer <token>`."""

    token: str


class AuthStatus(BaseModel):
    """Whether a password is needed, and whether the caller's token is valid."""

    required: bool
    authenticated: bool


@router.get("/status", response_model=AuthStatus)
def auth_status(request: Request, settings: SettingsDep) -> AuthStatus:
    """Lets the UI decide whether to show the password screen."""
    try:
        require_access(request, settings)
    except HTTPException:
        return AuthStatus(required=True, authenticated=False)
    return AuthStatus(required=settings.demo_password is not None, authenticated=True)


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, settings: SettingsDep) -> LoginResponse:
    """Check the shared demo password and return a token."""
    if not settings.demo_password:
        return LoginResponse(token="")
    if not password_matches(body.password, settings.demo_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Wrong password.")
    return LoginResponse(token=make_token(settings.demo_password))
