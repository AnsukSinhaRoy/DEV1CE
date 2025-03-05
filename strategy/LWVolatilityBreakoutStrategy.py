class DonchianMomentumStrategy:
    @staticmethod
    def run_strategy(df, i, prev_values):
        # Exit logic for existing position
        if prev_values['entered']:
            current_close = df.iloc[i]['close']
            # Check take profit condition
            if current_close >= prev_values.get('take_profit', 0):
                return -1, prev_values['stop_loss']
            return 0, prev_values['stop_loss']

        # Entry logic
        entry_conditions = {
            'price_touch': df.iloc[i]['high'] >= df.iloc[i]['DC_upper'],
            'lwti_green': df.iloc[i]['slowk'] > df.iloc[i]['slowd'],
            'volume_condition': (df.iloc[i]['volume'] > df.iloc[i]['Volume_Threshold']) 
                             and (df.iloc[i]['close'] > df.iloc[i]['open'])
        }

        if all(entry_conditions.values()):
            # Calculate stop loss
            dc_upper = df.iloc[i]['DC_upper']
            dc_middle = df.iloc[i]['DC_middle']
            atr = df.iloc[i]['ATR']

            if (dc_upper - dc_middle) > 2 * atr:
                # Get swing low from previous 5 periods (excluding current bar)
                start_idx = max(0, i - 5)
                swing_low = df.iloc[start_idx:i]['low'].min()
                stop_loss = swing_low
            else:
                stop_loss = dc_middle

            # Calculate take profit
            entry_price = df.iloc[i]['close']
            take_profit = entry_price + 2 * (entry_price - stop_loss)

            # Update state variables
            prev_values.update({
                'entry_price': entry_price,
                'take_profit': take_profit,
                'stop_loss': stop_loss
            })

            return 1, stop_loss

        return 0, 0