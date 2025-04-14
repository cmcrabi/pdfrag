from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import relationship
from app.database import Base

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(255), unique=True, index=True, nullable=False)

    # Relationship back to documents (optional, for querying from product)
    documents = relationship("Document", back_populates="product")

    def __repr__(self):
        return f"<Product {self.name}>" 