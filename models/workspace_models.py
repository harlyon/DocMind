import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import uuid
from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


def generate_uuid() -> str:
    return str(uuid.uuid4())


class Workspace(Base):
    """
    A workspace groups multiple related documents together.

    Example use cases:
      - All versions of a contract (v1, v2, v3)
      - Q1 + Q2 + Q3 financial reports
      - Multiple expert opinions on the same case

    Retrieval queries span ALL documents in the workspace,
    with each result chunk attributed to its source document.
    """
    __tablename__ = "workspaces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    user_id: Mapped[str] = mapped_column(String(36), index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Domain tag for template selection in the checklist feature
    # e.g. "legal", "finance", "hr", "general"
    domain: Mapped[str] = mapped_column(String(50), default="general")

    created_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, server_default=func.now(), onupdate=func.now()
    )

    # Relationships
    workspace_documents: Mapped[list["WorkspaceDocument"]] = relationship(
        back_populates="workspace",
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        return f"<Workspace id={self.id} name={self.name}>"


class WorkspaceDocument(Base):
    """
    Join table linking workspaces to documents.

    A document can belong to multiple workspaces.
    A workspace can contain multiple documents.

    The display_name lets users label documents within a workspace context,
    e.g. "Contract v1", "Contract v2" rather than the raw filename.
    """
    __tablename__ = "workspace_documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=generate_uuid)
    workspace_id: Mapped[str] = mapped_column(
        ForeignKey("workspaces.id"), index=True
    )
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id"), index=True
    )

    # Human-readable label for this doc within the workspace context
    display_name: Mapped[str | None] = mapped_column(String(255), nullable=True)

    added_at: Mapped[datetime] = mapped_column(DateTime, server_default=func.now())

    # Relationships
    workspace: Mapped["Workspace"] = relationship(back_populates="workspace_documents")
    document: Mapped["Document"] = relationship()  # type: ignore[name-defined]

    def __repr__(self) -> str:
        return (
            f"<WorkspaceDocument workspace={self.workspace_id} "
            f"document={self.document_id}>"
        )