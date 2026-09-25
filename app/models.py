from datetime import datetime, timedelta, timezone
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


def ist_now():
    """Returns the current time in India Standard Time (UTC+5:30)."""
    return datetime.now(timezone.utc).replace(tzinfo=None) + timedelta(hours=5, minutes=30)


class PredictionHistory(db.Model):
    """Stores every prediction made through the website, for history/tracking."""
    __tablename__ = "prediction_history"

    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, default=ist_now)

    # Clinical inputs
    follicle_r = db.Column(db.Float)
    follicle_l = db.Column(db.Float)
    skin_darkening = db.Column(db.Integer)
    hair_growth = db.Column(db.Integer)
    weight_gain = db.Column(db.Integer)
    cycle = db.Column(db.String(1))
    cycle_length = db.Column(db.Float)
    amh = db.Column(db.Float)
    prl = db.Column(db.Float)
    fsh_lh = db.Column(db.Float)
    fast_food = db.Column(db.Integer)
    pimples = db.Column(db.Integer)

    # Results
    prediction = db.Column(db.String(50))
    risk_score = db.Column(db.Float)
    clinical_risk_score = db.Column(db.Float)
    image_risk_score = db.Column(db.Float, nullable=True)
    modality_used = db.Column(db.String(100))
    phenotype = db.Column(db.String(100))

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.strftime("%Y-%m-%d %H:%M"),
            "prediction": self.prediction,
            "risk_score": self.risk_score,
            "clinical_risk_score": self.clinical_risk_score,
            "image_risk_score": self.image_risk_score,
            "modality_used": self.modality_used,
            "phenotype": self.phenotype,
        }
