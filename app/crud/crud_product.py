from sqlalchemy.orm import Session
from app.models.product import Product
from app.schemas.product import ProductCreate

def get_product(db: Session, product_id: int) -> Product | None:
    """Gets a product by its primary key ID."""
    return db.query(Product).filter(Product.id == product_id).first()

def get_product_by_name(db: Session, name: str) -> Product | None:
    return db.query(Product).filter(Product.name == name).first()

def create_product(db: Session, product: ProductCreate) -> Product:
    db_product = Product(name=product.name)
    db.add(db_product)
    db.commit()
    db.refresh(db_product)
    return db_product

def get_or_create_product(db: Session, name: str) -> Product:
    db_product = get_product_by_name(db=db, name=name)
    if db_product:
        return db_product
    return create_product(db=db, product=ProductCreate(name=name)) 