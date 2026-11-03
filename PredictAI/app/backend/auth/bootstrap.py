import os

from app.backend.auth.database import Base, SessionLocal, engine
from app.backend.auth.models import User
from app.backend.auth.security import hash_password


def initialize_auth():
    Base.metadata.create_all(bind=engine)

    username = os.getenv("ADMIN_USERNAME")
    password = os.getenv("ADMIN_PASSWORD")

    if not username or not password:
        print("WARNING: Admin credentials not configured.")
        print("Set ADMIN_USERNAME and ADMIN_PASSWORD before creating an admin.")
        return

    if len(password) < 12:
        raise RuntimeError("ADMIN_PASSWORD must be at least 12 characters.")

    db = SessionLocal()

    try:
        existing = db.query(User).filter(
            User.username == username
        ).first()

        if not existing:
            admin = User(
                username=username,
                password_hash=hash_password(password),
                is_admin=True,
                is_active=True
            )

            db.add(admin)
            db.commit()

            print(f"Initial administrator '{username}' created.")

        elif not existing.is_admin:
            raise RuntimeError(
                "Configured admin username already belongs to a non-admin user."
            )

    finally:
        db.close()