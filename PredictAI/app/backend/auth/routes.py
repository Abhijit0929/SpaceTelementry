from pydantic import BaseModel, Field
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.backend.auth.database import get_db
from app.backend.auth.models import User
from app.backend.auth.security import (
    create_access_token,
    get_current_user,
    hash_password,
    require_admin,
    verify_password,
)

router = APIRouter(prefix="/api/auth", tags=["Authentication"])


class LoginRequest(BaseModel):
    username: str
    password: str


class CreateUserRequest(BaseModel):
    username: str = Field(min_length=3, max_length=100)
    password: str = Field(min_length=12, max_length=128)


@router.post("/login")
def login(data: LoginRequest, db: Session = Depends(get_db)):
    user = db.query(User).filter(
        User.username == data.username
    ).first()

    if (
        not user
        or not user.is_active
        or not verify_password(data.password, user.password_hash)
    ):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid username or password"
        )

    token = create_access_token(user.username)

    return {
        "access_token": token,
        "token_type": "bearer",
        "username": user.username,
        "is_admin": user.is_admin
    }


@router.get("/me")
def me(current_user: User = Depends(get_current_user)):
    return {
        "username": current_user.username,
        "is_admin": current_user.is_admin
    }


@router.post("/users", status_code=status.HTTP_201_CREATED)
def create_user(
    data: CreateUserRequest,
    db: Session = Depends(get_db),
    admin: User = Depends(require_admin)
):
    existing = db.query(User).filter(
        User.username == data.username
    ).first()

    if existing:
        raise HTTPException(
            status_code=409,
            detail="Username already exists"
        )

    user = User(
        username=data.username,
        password_hash=hash_password(data.password),
        is_admin=False
    )

    db.add(user)
    db.commit()

    return {
        "message": "User account created",
        "username": user.username
    }